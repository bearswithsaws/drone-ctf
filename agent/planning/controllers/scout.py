"""Battery-safe frontier scouting for the live match loop.

Wire coordinates are relative to the acting unit. Ingest normalises those
observations into the command-centre-relative frame held by ``WorldModel``;
this controller plans only inside that shared local frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from agent.planning.pathfind import Pathfinder, PathfindingConfig, TileCost
from agent.planning.pipeliner import PlannedAction
from agent.planning.tasks import ScoutSector
from agent.rules.costs import estimate_battery_cost
from agent.rules.hexmath import get_neighbor, hex_distance_cube
from agent.sim.entity_sim import DroneSimState, EntitySim
from agent.transport.action_tracker import Precondition
from agent.world.model import DroneRecord, WorldModel
from agent.world.tiles import Coord, Terrain


class ScoutPhase(str, Enum):
    WAITING = "waiting"
    TRAVELLING = "travelling"
    SCANNING = "scanning"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ScoutConfig:
    # Match the strategist's emergency-charge trigger so a scout leg never
    # deliberately crosses below the point where normal work is pre-empted.
    reserve_fraction: float = 0.30
    minimum_reserve: int = 100
    max_leg_tiles: int = 4
    max_threat_cost: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.reserve_fraction < 1:
            raise ValueError("reserve_fraction must be in [0, 1)")
        if self.minimum_reserve < 0:
            raise ValueError("minimum_reserve must be non-negative")
        if self.max_leg_tiles < 1:
            raise ValueError("max_leg_tiles must be positive")
        if not math.isfinite(self.max_threat_cost) or self.max_threat_cost < 0:
            raise ValueError("max_threat_cost must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ScoutPlan:
    drone_id: str
    phase: ScoutPhase
    actions: tuple[PlannedAction, ...] = ()
    destination: Coord | None = None
    reason: str = ""


class ScoutController:
    """Advance over known-safe tiles to a frontier, then scan from there."""

    def __init__(
        self,
        world: WorldModel,
        sim: EntitySim,
        drone_id: str,
        task: ScoutSector,
        *,
        threat_cost: TileCost | None = None,
        pathfinding: PathfindingConfig | None = None,
        config: ScoutConfig | None = None,
    ) -> None:
        if not drone_id.strip():
            raise ValueError("drone_id cannot be empty")
        self._world = world
        self._sim = sim
        self.drone_id = drone_id
        self.task = task
        self._threat_cost = threat_cost
        self._pathfinding = pathfinding
        self.config = config or ScoutConfig()

    def plan(self) -> ScoutPlan:
        drone = self._world.get_drone(self.drone_id)
        missing = self._missing_state(drone)
        if missing:
            return ScoutPlan(self.drone_id, ScoutPhase.WAITING, reason=missing)
        state = self._sim.get_drone(self.drone_id)
        assert drone is not None and drone.coord is not None and drone.direction is not None
        assert state is not None

        safe = self._known_safe_tiles(drone.coord)
        frontiers = self._frontiers(safe)
        if not frontiers:
            return ScoutPlan(
                self.drone_id,
                ScoutPhase.BLOCKED,
                destination=drone.coord,
                reason="sector has no remaining known-safe frontier",
            )
        routes = self._reachable_routes(drone, frontiers, safe)
        if not routes:
            return ScoutPlan(
                self.drone_id,
                ScoutPhase.BLOCKED,
                destination=drone.coord,
                reason="known-safe frontiers are unreachable within one scout leg",
            )

        for destination, route in self._ranked_routes(drone.coord, routes):
            movement = route.compile(
                drone.direction,
                level=self._equipment_level("propulsion"),
                precondition_factory=self._motion_guard,
            )
            actions = (*movement, self._scan_action(destination))
            if not self._preserves_reserve(state, actions):
                continue
            distance = float(hex_distance_cube(0, 0, *destination))
            actions = tuple(
                PlannedAction(
                    action.action,
                    action.payload,
                    precondition=action.precondition,
                    distance=distance,
                    endpoint=action.endpoint,
                )
                for action in actions
            )
            return ScoutPlan(
                self.drone_id,
                ScoutPhase.TRAVELLING if movement else ScoutPhase.SCANNING,
                actions,
                destination,
            )

        return ScoutPlan(
            self.drone_id,
            ScoutPhase.BLOCKED,
            reason="frontier leg would violate battery reserve",
        )

    def _known_safe_tiles(self, current: Coord) -> set[Coord]:
        occupied = {
            drone.coord
            for drone in self._world.drones()
            if drone.drone_id != self.drone_id and drone.coord is not None
        }
        safe: set[Coord] = set()
        for tile in self._world.tiles():
            if tile.terrain_type not in (Terrain.NORMAL, Terrain.DIFFICULT):
                continue
            if tile.has_building or tile.coord in occupied:
                continue
            if tile.has_drone and tile.coord != current:
                continue
            if self._threat_at(tile.coord) > self.config.max_threat_cost:
                continue
            safe.add(tile.coord)
        # An initial status can arrive before a terrain scan. The tile under
        # the drone is necessarily enterable because the drone occupies it.
        safe.add(current)
        return safe

    def _frontiers(self, safe: set[Coord]) -> list[Coord]:
        return [
            coord
            for coord in safe
            if hex_distance_cube(*self.task.center, *coord) <= self.task.radius
            and any(self._is_unknown(get_neighbor(*coord, direction)) for direction in range(6))
        ]

    def _reachable_routes(self, drone: DroneRecord, candidates: list[Coord], safe: set[Coord]):
        assert drone.coord is not None
        finder = Pathfinder(
            self._world,
            threat_cost=self._threat_cost,
            config=self._pathfinding,
        )
        routes = []
        for destination in candidates:
            route = finder.find_path(drone.coord, destination, allowed=safe)
            if route is None or len(route.path) - 1 > self.config.max_leg_tiles:
                continue
            routes.append((destination, route))
        return routes

    def _ranked_routes(self, current: Coord, routes):
        moving = [item for item in routes if item[0] != current]
        candidates = moving or routes

        def rank(item) -> tuple[int, int, int, float, int, int]:
            destination, route = item
            tile = self._world.get_tile(destination)
            unknown_neighbors = sum(
                self._is_unknown(get_neighbor(*destination, direction)) for direction in range(6)
            )
            return (
                -hex_distance_cube(*self.task.center, *destination),
                -unknown_neighbors,
                tile.last_seen if tile is not None else -1,
                route.cost,
                destination[0],
                destination[1],
            )

        return sorted(candidates, key=rank)

    def _scan_action(self, position: Coord) -> PlannedAction:
        return PlannedAction(
            "sensors/scan",
            {"level": self._equipment_level("sensors")},
            precondition=self._scan_guard(position),
            distance=float(hex_distance_cube(0, 0, *position)),
        )

    def _preserves_reserve(
        self, state: DroneSimState, actions: tuple[PlannedAction, ...]
    ) -> bool:
        subjects = {
            "propulsion/drive": "Drive completed",
            "propulsion/turn": "Turn completed",
            "sensors/scan": "Scan completed",
        }
        cost = sum(
            estimate_battery_cost(
                subjects.get(action.action, action.action),
                action.payload,
                research_tiers=self._sim.research_tiers,
                total_weight=state.total_weight,
            )
            for action in actions
        )
        reserve = max(
            self.config.minimum_reserve,
            math.ceil(state.max_battery * self.config.reserve_fraction),
        )
        return state.current_battery - cost >= reserve

    def _is_unknown(self, coord: Coord) -> bool:
        tile = self._world.get_tile(coord)
        return tile is None or tile.terrain_type is None

    def _threat_at(self, coord: Coord) -> float:
        return float(self._threat_cost(coord)) if self._threat_cost is not None else 0.0

    def _equipment_level(self, equipment: str) -> int:
        state = self._sim.get_drone(self.drone_id)
        if state is None:
            return 1
        return min(3, max(1, state.equipment_levels.get(equipment, 1)))

    def _motion_guard(self, position: Coord, heading: int) -> Precondition:
        def current() -> bool:
            drone = self._world.get_drone(self.drone_id)
            return drone is not None and drone.coord == position and drone.direction == heading

        return current

    def _scan_guard(self, position: Coord) -> Precondition:
        def current() -> bool:
            drone = self._world.get_drone(self.drone_id)
            state = self._sim.get_drone(self.drone_id)
            return (
                drone is not None
                and drone.coord == position
                and state is not None
                and "sensors" in state.functional_equipment
            )

        return current

    def _missing_state(self, drone: DroneRecord | None) -> str:
        if drone is None or drone.coord is None:
            return "waiting for drone position checkpoint"
        if drone.direction is None:
            return "waiting for drone heading checkpoint"
        state = self._sim.get_drone(self.drone_id)
        if state is None:
            return "waiting for drone equipment/battery checkpoint"
        if state.max_battery < 1:
            return "waiting for a valid battery checkpoint"
        if "sensors" not in state.functional_equipment:
            return "drone has no functional sensors"
        if "propulsion" not in state.functional_equipment:
            return "drone has no functional propulsion"
        return ""


__all__ = ["ScoutConfig", "ScoutController", "ScoutPhase", "ScoutPlan"]
