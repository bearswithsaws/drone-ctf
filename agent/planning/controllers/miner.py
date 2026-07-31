"""Threat-aware, battery-safe planning for autonomous miner drones.

The miner is deliberately a replanning controller.  Each call to
:meth:`MinerController.plan` reads the latest world and entity-simulator
checkpoints, then emits the next safe leg of the mine/unload/charge loop.  A
leg targets exactly one server entity: drone movement/mining uses the drone
queue, while unloading and station charging use the relevant building queue.
This prevents a building action from racing ahead of a drone that is still in
transit.
"""

from __future__ import annotations

import math
from collections.abc import Container, Iterable
from dataclasses import dataclass
from enum import Enum

from agent.planning.pathfind import (
    AllowedTiles,
    MovementProfile,
    PathResult,
    Pathfinder,
    PathfindingConfig,
    TileCost,
)
from agent.planning.pipeliner import PlannedAction
from agent.planning.tasks import MineLoop
from agent.rules.costs import CHARGE_RATE, estimate_battery_cost
from agent.rules.hexmath import get_neighbor, hex_distance_cube
from agent.sim.entity_sim import DroneSimState, EntitySim
from agent.transport.action_tracker import EntityKind, Precondition
from agent.world.model import BuildingRecord, DroneRecord, WorldModel
from agent.world.tiles import Coord, TileBelief


class MinerPhase(str, Enum):
    """Observable state of one miner replanning decision."""

    WAITING = "waiting"
    SEEK_RESOURCE = "seek_resource"
    MINING = "mining"
    RETURNING = "returning"
    UNLOADING = "unloading"
    SEEK_CHARGER = "seek_charger"
    CHARGING = "charging"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class MinerConfig:
    """Safety and action-level policy for a miner."""

    mine_level: int = 1
    move_level: int = 1
    efficiency: float = 1.0
    reserve_fraction: float = 0.15
    minimum_reserve: int = 50
    charge_target_fraction: float = 0.9
    cargo_unit_weight: float = 1.0
    charger_id: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.mine_level <= 3:
            raise ValueError("mine_level must be between 1 and 3")
        if not 1 <= self.move_level <= 3:
            raise ValueError("move_level must be between 1 and 3")
        if not 0.01 <= self.efficiency <= 1.0:
            raise ValueError("efficiency must be between 0.01 and 1.0")
        if not 0 <= self.reserve_fraction <= 1:
            raise ValueError("reserve_fraction must be between 0 and 1")
        if self.minimum_reserve < 0:
            raise ValueError("minimum_reserve cannot be negative")
        if not 0 < self.charge_target_fraction <= 1:
            raise ValueError("charge_target_fraction must be in (0, 1]")
        if not math.isfinite(self.cargo_unit_weight) or self.cargo_unit_weight < 0:
            raise ValueError("cargo_unit_weight must be finite and non-negative")
        if self.charger_id is not None and not self.charger_id.strip():
            raise ValueError("charger_id cannot be empty")


@dataclass(frozen=True, slots=True)
class RoutedAction:
    """A planned action paired with the server entity queue that owns it."""

    entity_kind: EntityKind
    entity_id: str
    action: PlannedAction


@dataclass(frozen=True, slots=True)
class MinerPlan:
    """One coherent miner leg, ready to install in a pipeliner."""

    drone_id: str
    phase: MinerPhase
    actions: tuple[RoutedAction, ...] = ()
    destination: Coord | None = None
    resource_node: Coord | None = None
    required_battery: int = 0
    reserve_battery: int = 0
    reason: str = ""

    @property
    def entity_key(self) -> tuple[EntityKind, str] | None:
        """Return the single queue targeted by this leg, if it has actions."""

        if not self.actions:
            return None
        first = self.actions[0]
        return (first.entity_kind, first.entity_id)

    def actions_for(
        self, entity_kind: EntityKind | str, entity_id: str
    ) -> tuple[PlannedAction, ...]:
        """Extract controller output suitable for ``Pipeliner.register``."""

        kind = EntityKind(entity_kind)
        return tuple(
            routed.action
            for routed in self.actions
            if routed.entity_kind is kind and routed.entity_id == entity_id
        )

    def __post_init__(self) -> None:
        keys = {(item.entity_kind, item.entity_id) for item in self.actions}
        if len(keys) > 1:
            raise ValueError("a miner leg cannot target multiple server queues")


@dataclass(frozen=True, slots=True)
class _Route:
    result: PathResult
    actions: tuple[PlannedAction, ...]
    destination: Coord
    end_heading: int
    battery_cost: int


class MinerController:
    """Plan a continuous mine -> unload -> charge loop for one drone.

    Callers should request a new plan after the current leg is exhausted or
    invalidated by a world change.  The returned routed actions make the queue
    transition explicit, so a fleet coordinator can serialize two miners that
    share the same refiner or charging station.
    """

    def __init__(
        self,
        world: WorldModel,
        sim: EntitySim,
        drone_id: str,
        task: MineLoop,
        *,
        threat_cost: TileCost | None = None,
        comms_risk: TileCost | None = None,
        pathfinding: PathfindingConfig | None = None,
        movement: MovementProfile | None = None,
        allowed: AllowedTiles | None = None,
        config: MinerConfig | None = None,
    ) -> None:
        if not drone_id.strip():
            raise ValueError("drone_id cannot be empty")
        self._world = world
        self._sim = sim
        self.drone_id = drone_id
        self.task = task
        self.config = config or MinerConfig()
        self._threat_cost = threat_cost
        self._comms_risk = comms_risk
        self._pathfinding = pathfinding
        self._movement = movement
        self._allowed = allowed

    def plan(self) -> MinerPlan:
        """Return the next safe leg from the latest observed state."""

        drone = self._world.get_drone(self.drone_id)
        state = self._sim.get_drone(self.drone_id)
        missing = self._missing_state(drone, state)
        if missing:
            return self._plan(MinerPhase.WAITING, reason=missing)
        assert drone is not None and drone.coord is not None and drone.direction is not None
        assert state is not None

        dropoff = self._world.get_building(self.task.dropoff_id)
        if dropoff is None or not _building_tiles(dropoff):
            return self._plan(
                MinerPhase.BLOCKED,
                reason=f"dropoff {self.task.dropoff_id!r} has no known footprint",
            )

        finder = self._new_pathfinder()
        reserve = max(
            self.config.minimum_reserve,
            math.ceil(state.max_battery * self.config.reserve_fraction),
        )
        effective_weight = state.total_weight + state.hopper_load * self.config.cargo_unit_weight

        # A partially loaded drone already next to a viable node should keep
        # mining until the hopper fills, provided the return margin remains.
        resource = self._select_resource(finder, drone, effective_weight)
        if state.hopper_load and resource is not None and self._can_mine_here(drone, resource):
            if state.hopper_capacity > 0 and state.hopper_load < state.hopper_capacity:
                return self._plan_mine_or_charge(
                    finder,
                    drone,
                    state,
                    resource,
                    dropoff,
                    reserve,
                    effective_weight,
                )

        if state.hopper_load:
            return self._plan_return_or_charge(
                finder,
                drone,
                state,
                dropoff,
                reserve,
                effective_weight,
                resource.coord if resource is not None else None,
            )

        if state.hopper_capacity < 1:
            return self._plan(
                MinerPhase.BLOCKED,
                reserve=reserve,
                reason="drone has no checkpointed hopper capacity",
            )
        if resource is None:
            return self._plan(
                MinerPhase.BLOCKED,
                reserve=reserve,
                reason="no viable resource node is known",
            )
        return self._plan_mine_or_charge(
            finder,
            drone,
            state,
            resource,
            dropoff,
            reserve,
            effective_weight,
        )

    def _plan_mine_or_charge(
        self,
        finder: Pathfinder,
        drone: DroneRecord,
        state: DroneSimState,
        resource: TileBelief,
        dropoff: BuildingRecord,
        reserve: int,
        effective_weight: float,
    ) -> MinerPlan:
        assert drone.coord is not None and drone.direction is not None
        outbound = self._route(
            finder,
            drone.coord,
            drone.direction,
            _adjacent_tiles((resource.coord,)),
            effective_weight,
        )
        if outbound is None:
            return self._plan(
                MinerPhase.BLOCKED,
                resource=resource.coord,
                reserve=reserve,
                reason=f"resource node {resource.coord!r} is unreachable",
            )

        dropoff_route = self._route(
            finder,
            outbound.destination,
            outbound.end_heading,
            _service_tiles(dropoff),
            effective_weight,
        )
        if dropoff_route is None:
            return self._plan(
                MinerPhase.BLOCKED,
                resource=resource.coord,
                reserve=reserve,
                reason=f"dropoff {dropoff.building_id!r} is unreachable from resource",
            )
        charger_route = self._nearest_charger_route(
            finder,
            dropoff_route.destination,
            dropoff_route.end_heading,
            effective_weight,
        )
        if charger_route is None:
            return self._plan(
                MinerPhase.BLOCKED,
                resource=resource.coord,
                reserve=reserve,
                reason="no reachable charging station is known",
            )

        mine_cost = estimate_battery_cost(
            "mine completed",
            {"level": self.config.mine_level},
            research_tiers=self._sim.research_tiers,
            total_weight=effective_weight,
        )
        required = (
            outbound.battery_cost
            + mine_cost
            + dropoff_route.battery_cost
            + charger_route[1].battery_cost
            + reserve
        )
        if state.current_battery < required:
            return self._plan_charge_or_travel(
                finder,
                drone,
                state,
                required,
                reserve,
                effective_weight,
                resource.coord,
            )
        if outbound.actions:
            return self._movement_plan(
                MinerPhase.SEEK_RESOURCE,
                outbound,
                resource.coord,
                required,
                reserve,
            )

        mine = PlannedAction(
            "drill/mine",
            {
                "level": self.config.mine_level,
                "resource_q": resource.q,
                "resource_r": resource.r,
            },
            precondition=self._mine_guard(resource.coord, outbound.destination),
            distance=_distance_from_base(outbound.destination),
        )
        return self._plan(
            MinerPhase.MINING,
            actions=(RoutedAction(EntityKind.DRONE, self.drone_id, mine),),
            destination=outbound.destination,
            resource=resource.coord,
            required=required,
            reserve=reserve,
            reason="mine one batch, then re-evaluate hopper and battery",
        )

    def _plan_return_or_charge(
        self,
        finder: Pathfinder,
        drone: DroneRecord,
        state: DroneSimState,
        dropoff: BuildingRecord,
        reserve: int,
        effective_weight: float,
        resource: Coord | None,
    ) -> MinerPlan:
        assert drone.coord is not None and drone.direction is not None
        dropoff_route = self._route(
            finder,
            drone.coord,
            drone.direction,
            _service_tiles(dropoff),
            effective_weight,
        )
        if dropoff_route is None:
            return self._plan(
                MinerPhase.BLOCKED,
                resource=resource,
                reserve=reserve,
                reason=f"dropoff {dropoff.building_id!r} is unreachable",
            )
        charger_route = self._nearest_charger_route(
            finder,
            dropoff_route.destination,
            dropoff_route.end_heading,
            effective_weight,
        )
        if charger_route is None:
            return self._plan(
                MinerPhase.BLOCKED,
                resource=resource,
                reserve=reserve,
                reason="no reachable charging station is known",
            )
        required = dropoff_route.battery_cost + charger_route[1].battery_cost + reserve
        # Unloading is a building action and spends no drone battery.  If the
        # miner is already in range, feed the refiner before considering the
        # next charger leg; otherwise preserve enough margin for both legs.
        if dropoff_route.actions and state.current_battery < required:
            return self._plan_charge_or_travel(
                finder,
                drone,
                state,
                required,
                reserve,
                effective_weight,
                resource,
            )
        if dropoff_route.actions:
            return self._movement_plan(
                MinerPhase.RETURNING,
                dropoff_route,
                resource,
                required,
                reserve,
            )

        payload: dict[str, object] = {
            "q": dropoff_route.destination[0],
            "r": dropoff_route.destination[1],
            "efficiency": self.config.efficiency,
        }
        if self.task.resource_type is not None:
            payload["resource_type"] = self.task.resource_type
        unload = PlannedAction(
            "common/unload",
            payload,
            precondition=self._unload_guard(dropoff_route.destination),
            distance=_distance_from_base(dropoff_route.destination),
        )
        return self._plan(
            MinerPhase.UNLOADING,
            actions=(RoutedAction(EntityKind.BUILDING, dropoff.building_id, unload),),
            destination=dropoff_route.destination,
            resource=resource,
            required=required,
            reserve=reserve,
            reason="unload hopper into dropoff, then replan",
        )

    def _plan_charge_or_travel(
        self,
        finder: Pathfinder,
        drone: DroneRecord,
        state: DroneSimState,
        required: int,
        reserve: int,
        effective_weight: float,
        resource: Coord | None,
    ) -> MinerPlan:
        assert drone.coord is not None and drone.direction is not None
        selected = self._nearest_charger_route(
            finder,
            drone.coord,
            drone.direction,
            effective_weight,
        )
        if selected is None:
            return self._plan(
                MinerPhase.BLOCKED,
                resource=resource,
                required=required,
                reserve=reserve,
                reason="no reachable charging station is known",
            )
        charger, route = selected
        if route.actions:
            minimum_arrival = route.battery_cost + reserve
            if state.current_battery < minimum_arrival:
                return self._plan(
                    MinerPhase.BLOCKED,
                    resource=resource,
                    required=required,
                    reserve=reserve,
                    reason=(
                        f"battery {state.current_battery} is below the safe charger leg "
                        f"requirement {minimum_arrival}"
                    ),
                )
            return self._movement_plan(
                MinerPhase.SEEK_CHARGER,
                route,
                resource,
                required,
                reserve,
            )

        target = min(
            state.max_battery,
            max(required, math.ceil(state.max_battery * self.config.charge_target_fraction)),
        )
        if state.current_battery >= target:
            return self._plan(
                MinerPhase.BLOCKED,
                destination=route.destination,
                resource=resource,
                required=required,
                reserve=reserve,
                reason=(
                    f"full battery capacity {state.max_battery} cannot satisfy required "
                    f"mission margin {required}"
                ),
            )
        count = math.ceil((target - state.current_battery) / CHARGE_RATE)
        charge_actions = tuple(
            RoutedAction(
                EntityKind.BUILDING,
                charger.building_id,
                PlannedAction(
                    "charging_station/charge",
                    {
                        "q": route.destination[0],
                        "r": route.destination[1],
                        "efficiency": self.config.efficiency,
                    },
                    precondition=self._charge_guard(route.destination, target),
                    distance=_distance_from_base(route.destination),
                ),
            )
            for _ in range(count)
        )
        return self._plan(
            MinerPhase.CHARGING,
            actions=charge_actions,
            destination=route.destination,
            resource=resource,
            required=required,
            reserve=reserve,
            reason=f"charge to at least {target} battery",
        )

    def _movement_plan(
        self,
        phase: MinerPhase,
        route: _Route,
        resource: Coord | None,
        required: int,
        reserve: int,
    ) -> MinerPlan:
        actions = tuple(
            RoutedAction(EntityKind.DRONE, self.drone_id, action) for action in route.actions
        )
        return self._plan(
            phase,
            actions=actions,
            destination=route.destination,
            resource=resource,
            required=required,
            reserve=reserve,
        )

    def _select_resource(
        self,
        finder: Pathfinder,
        drone: DroneRecord,
        effective_weight: float,
    ) -> TileBelief | None:
        assert drone.coord is not None and drone.direction is not None
        by_coord = {
            tile.coord: tile
            for tile in self._world.resource_nodes()
            if (self.task.resource_type is None or tile.resource_type == self.task.resource_type)
            and (tile.resource_amount is None or tile.resource_amount > 0)
        }
        assigned = self._world.get_tile(self.task.resource_node)
        if assigned is None:
            assigned = TileBelief(
                q=self.task.resource_node[0],
                r=self.task.resource_node[1],
                has_resource=True,
                resource_type=self.task.resource_type,
            )
        if assigned.has_resource and (
            self.task.resource_type is None or assigned.resource_type == self.task.resource_type
        ):
            by_coord[assigned.coord] = assigned

        best: tuple[float, int, TileBelief] | None = None
        for tile in by_coord.values():
            route = self._route(
                finder,
                drone.coord,
                drone.direction,
                _adjacent_tiles((tile.coord,)),
                effective_weight,
            )
            if route is None:
                continue
            assigned_rank = 0 if tile.coord == self.task.resource_node else 1
            candidate = (route.result.cost, assigned_rank, tile)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        return best[2] if best is not None else None

    def _nearest_charger_route(
        self,
        finder: Pathfinder,
        start: Coord,
        heading: int,
        effective_weight: float,
    ) -> tuple[BuildingRecord, _Route] | None:
        chargers = [
            building
            for building in self._world.buildings()
            if building.building_type == "drone_charging_station"
            and (self.config.charger_id is None or building.building_id == self.config.charger_id)
            and _building_tiles(building)
        ]
        best: tuple[float, str, BuildingRecord, _Route] | None = None
        for charger in chargers:
            route = self._route(
                finder,
                start,
                heading,
                _service_tiles(charger),
                effective_weight,
            )
            if route is None:
                continue
            candidate = (route.result.cost, charger.building_id, charger, route)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        return (best[2], best[3]) if best is not None else None

    def _route(
        self,
        finder: Pathfinder,
        start: Coord,
        heading: int,
        destinations: Container[Coord],
        effective_weight: float,
    ) -> _Route | None:
        best: tuple[float, int, Coord, PathResult, tuple[PlannedAction, ...], int] | None = None
        for destination in sorted(destinations):
            result = finder.find_path(start, destination, allowed=self._allowed)
            if result is None:
                continue
            actions = result.compile(
                heading,
                level=self.config.move_level,
                precondition_factory=self._motion_guard,
            )
            battery = sum(
                estimate_battery_cost(
                    f"{action.action} completed",
                    action.payload,
                    research_tiers=self._sim.research_tiers,
                    total_weight=effective_weight,
                )
                for action in actions
            )
            end_heading = _heading_after(heading, actions)
            candidate = (
                result.cost,
                battery,
                destination,
                result,
                actions,
                end_heading,
            )
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        if best is None:
            return None
        distance = _distance_from_base(best[2])
        actions = tuple(
            PlannedAction(
                action.action,
                action.payload,
                precondition=action.precondition,
                distance=distance,
                endpoint=action.endpoint,
            )
            for action in best[4]
        )
        return _Route(best[3], actions, best[2], best[5], best[1])

    def _new_pathfinder(self) -> Pathfinder:
        return Pathfinder(
            self._world,
            threat_cost=self._threat_cost,
            comms_risk=self._comms_risk,
            config=self._pathfinding,
            movement=self._movement,
        )

    def _motion_guard(self, position: Coord, heading: int) -> Precondition:
        def current() -> bool:
            drone = self._world.get_drone(self.drone_id)
            return drone is not None and drone.coord == position and drone.direction == heading

        return current

    def _mine_guard(self, resource: Coord, position: Coord) -> Precondition:
        def current() -> bool:
            drone = self._world.get_drone(self.drone_id)
            state = self._sim.get_drone(self.drone_id)
            tile = self._world.get_tile(resource)
            resource_present = tile is None or (
                tile.has_resource and (tile.resource_amount is None or tile.resource_amount > 0)
            )
            return (
                drone is not None
                and drone.coord == position
                and state is not None
                and state.hopper_capacity > state.hopper_load
                and resource_present
            )

        return current

    def _unload_guard(self, position: Coord) -> Precondition:
        def current() -> bool:
            drone = self._world.get_drone(self.drone_id)
            state = self._sim.get_drone(self.drone_id)
            return (
                drone is not None
                and drone.coord == position
                and state is not None
                and state.hopper_load > 0
            )

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

    @staticmethod
    def _missing_state(drone: DroneRecord | None, state: DroneSimState | None) -> str:
        if drone is None or drone.coord is None:
            return "waiting for drone position checkpoint"
        if drone.direction is None:
            return "waiting for drone heading checkpoint"
        if state is None:
            return "waiting for drone battery/hopper checkpoint"
        if state.max_battery < 1:
            return "waiting for a valid battery checkpoint"
        return ""

    @staticmethod
    def _can_mine_here(drone: DroneRecord, resource: TileBelief) -> bool:
        return drone.coord in _adjacent_tiles((resource.coord,))

    def _plan(
        self,
        phase: MinerPhase,
        *,
        actions: tuple[RoutedAction, ...] = (),
        destination: Coord | None = None,
        resource: Coord | None = None,
        required: int = 0,
        reserve: int = 0,
        reason: str = "",
    ) -> MinerPlan:
        return MinerPlan(
            drone_id=self.drone_id,
            phase=phase,
            actions=actions,
            destination=destination,
            resource_node=resource,
            required_battery=required,
            reserve_battery=reserve,
            reason=reason,
        )


def group_miner_actions(
    plans: Iterable[MinerPlan],
) -> dict[tuple[EntityKind, str], tuple[PlannedAction, ...]]:
    """Combine concurrent miner legs into one output stream per server queue.

    Drone routes naturally remain separate.  Shared refiner and charging
    station actions are serialized in input order, allowing the fleet runtime
    to register one legal building plan for both starting miners.
    """

    grouped: dict[tuple[EntityKind, str], list[PlannedAction]] = {}
    for plan in plans:
        for routed in plan.actions:
            grouped.setdefault((routed.entity_kind, routed.entity_id), []).append(routed.action)
    return {key: tuple(actions) for key, actions in grouped.items()}


def _building_tiles(building: BuildingRecord) -> tuple[Coord, ...]:
    if building.tiles:
        return building.tiles
    return (building.origin,) if building.origin is not None else ()


def _adjacent_tiles(tiles: tuple[Coord, ...]) -> frozenset[Coord]:
    occupied = set(tiles)
    return frozenset(
        get_neighbor(*tile, direction)
        for tile in tiles
        for direction in range(6)
        if get_neighbor(*tile, direction) not in occupied
    )


def _service_tiles(building: BuildingRecord) -> frozenset[Coord]:
    return _adjacent_tiles(_building_tiles(building))


def _heading_after(initial: int, actions: tuple[PlannedAction, ...]) -> int:
    heading = initial
    for action in actions:
        if action.action == "propulsion/turn":
            heading = (heading + int(action.payload.get("direction", 0))) % 6
    return heading


def _distance_from_base(coord: Coord) -> float:
    return float(hex_distance_cube(0, 0, *coord))


__all__ = [
    "group_miner_actions",
    "MinerConfig",
    "MinerController",
    "MinerPhase",
    "MinerPlan",
    "RoutedAction",
]
