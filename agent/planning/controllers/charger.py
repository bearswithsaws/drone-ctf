"""Role-independent emergency charging controller."""

from __future__ import annotations

import math
from dataclasses import dataclass

from agent.planning.controllers.miner import MinerPhase, MinerPlan, RoutedAction
from agent.planning.pathfind import Pathfinder, TileCost
from agent.planning.pipeliner import PlannedAction
from agent.planning.tasks import Recharge
from agent.rules.costs import CHARGE_RATE, estimate_battery_cost
from agent.rules.hexmath import abs_to_rel, get_neighbor, hex_distance_cube
from agent.sim.entity_sim import EntitySim
from agent.transport.action_tracker import EntityKind, Precondition
from agent.world.model import BuildingRecord, WorldModel
from agent.world.tiles import Coord, Terrain


@dataclass(frozen=True, slots=True)
class ChargerConfig:
    minimum_arrival_reserve: int = 25
    efficiency: float = 1.0

    def __post_init__(self) -> None:
        if self.minimum_arrival_reserve < 0:
            raise ValueError("minimum_arrival_reserve must be non-negative")
        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency must be in (0, 1]")


class ChargerController:
    """Route one specifically assigned drone to a station and charge it.

    Movement is restricted to observed traversable, unoccupied, threat-free
    tiles. The station action uses coordinates relative to the station, which
    is the acting unit for that building-owned command.
    """

    def __init__(
        self,
        world: WorldModel,
        sim: EntitySim,
        drone_id: str,
        task: Recharge,
        *,
        threat_cost: TileCost | None = None,
        config: ChargerConfig | None = None,
    ) -> None:
        self._world = world
        self._sim = sim
        self.drone_id = drone_id
        self.task = task
        self._threat_cost = threat_cost
        self.config = config or ChargerConfig()

    def plan(self) -> MinerPlan:
        drone = self._world.get_drone(self.drone_id)
        state = self._sim.get_drone(self.drone_id)
        if drone is None or drone.coord is None or drone.direction is None or state is None:
            return MinerPlan(
                self.drone_id,
                MinerPhase.WAITING,
                reason="waiting for drone position, heading, and battery checkpoint",
            )
        if state.max_battery < 1:
            return MinerPlan(
                self.drone_id,
                MinerPhase.WAITING,
                reason="waiting for a valid battery checkpoint",
            )

        selected = self._nearest_route(drone.coord, drone.direction, state.total_weight)
        if selected is None:
            return MinerPlan(
                self.drone_id,
                MinerPhase.BLOCKED,
                reason="no known-safe route to a charging station",
            )
        charger, destination, actions, battery_cost = selected
        target = math.ceil(state.max_battery * self.task.target_fraction)

        if actions:
            required = battery_cost + self.config.minimum_arrival_reserve
            if state.current_battery < required:
                return MinerPlan(
                    self.drone_id,
                    MinerPhase.BLOCKED,
                    destination=destination,
                    required_battery=required,
                    reserve_battery=self.config.minimum_arrival_reserve,
                    reason=(
                        f"battery {state.current_battery} cannot safely reach charger; "
                        f"requires {required}"
                    ),
                )
            return MinerPlan(
                self.drone_id,
                MinerPhase.SEEK_CHARGER,
                tuple(
                    RoutedAction(EntityKind.DRONE, self.drone_id, action)
                    for action in actions
                ),
                destination=destination,
                required_battery=required,
                reserve_battery=self.config.minimum_arrival_reserve,
            )

        if state.current_battery >= target:
            return MinerPlan(
                self.drone_id,
                MinerPhase.WAITING,
                destination=destination,
                reason=f"charging recovery target {target} reached",
            )

        origin = charger.origin or charger.tiles[0]
        relative_q, relative_r = abs_to_rel(*destination, *origin)
        count = math.ceil((target - state.current_battery) / CHARGE_RATE)
        charges = tuple(
            RoutedAction(
                EntityKind.BUILDING,
                charger.building_id,
                PlannedAction(
                    "charging_station/charge",
                    {
                        "q": relative_q,
                        "r": relative_r,
                        "efficiency": self.config.efficiency,
                    },
                    precondition=self._charge_guard(destination, target),
                    distance=float(hex_distance_cube(0, 0, *destination)),
                ),
            )
            for _ in range(count)
        )
        return MinerPlan(
            self.drone_id,
            MinerPhase.CHARGING,
            charges,
            destination=destination,
            required_battery=target,
            reserve_battery=self.config.minimum_arrival_reserve,
            reason=f"charge to at least {target}",
        )

    def _nearest_route(
        self, start: Coord, heading: int, total_weight: float
    ) -> tuple[BuildingRecord, Coord, tuple[PlannedAction, ...], int] | None:
        safe = self._known_safe(start)
        finder = Pathfinder(self._world, threat_cost=self._threat_cost)
        best = None
        for charger in self._world.buildings():
            if charger.building_type != "drone_charging_station":
                continue
            tiles = charger.tiles or ((charger.origin,) if charger.origin is not None else ())
            for destination in sorted(_service_tiles(tiles)):
                if destination not in safe:
                    continue
                route = finder.find_path(start, destination, allowed=safe)
                if route is None:
                    continue
                actions = route.compile(
                    heading,
                    level=1,
                    precondition_factory=self._motion_guard,
                )
                battery = sum(
                    estimate_battery_cost(
                        f"{action.action} completed",
                        action.payload,
                        research_tiers=self._sim.research_tiers,
                        total_weight=total_weight,
                    )
                    for action in actions
                )
                candidate = (route.cost, battery, charger.building_id, charger, destination, actions)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        if best is None:
            return None
        return best[3], best[4], best[5], best[1]

    def _known_safe(self, current: Coord) -> set[Coord]:
        occupied = {
            drone.coord
            for drone in self._world.drones()
            if drone.drone_id != self.drone_id and drone.coord is not None
        }
        safe = {
            tile.coord
            for tile in self._world.tiles()
            if tile.terrain_type in (Terrain.NORMAL, Terrain.DIFFICULT)
            and not tile.has_building
            and tile.coord not in occupied
            and (not tile.has_drone or tile.coord == current)
            and self._threat_at(tile.coord) == 0.0
        }
        safe.add(current)
        return safe

    def _threat_at(self, coord: Coord) -> float:
        return float(self._threat_cost(coord)) if self._threat_cost is not None else 0.0

    def _motion_guard(self, position: Coord, heading: int) -> Precondition:
        def current() -> bool:
            drone = self._world.get_drone(self.drone_id)
            return drone is not None and drone.coord == position and drone.direction == heading

        return current

    def _charge_guard(self, position: Coord, target: int) -> Precondition:
        def current() -> bool:
            drone = self._world.get_drone(self.drone_id)
            state = self._sim.get_drone(self.drone_id)
            return (
                drone is not None
                and drone.coord == position
                and state is not None
                and state.current_battery < target
            )

        return current


def _service_tiles(tiles: tuple[Coord, ...]) -> set[Coord]:
    occupied = set(tiles)
    return {
        get_neighbor(*tile, direction)
        for tile in tiles
        for direction in range(6)
        if get_neighbor(*tile, direction) not in occupied
    }


__all__ = ["ChargerConfig", "ChargerController"]
