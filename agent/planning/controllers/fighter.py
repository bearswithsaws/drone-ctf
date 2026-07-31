"""Threat-aware engagement micro for laser and auto-cannon fighters.

The controller is intentionally replanning-oriented: every plan is derived
from the latest own-drone checkpoint and enemy tracks.  Movement is emitted as
one path leg; once the drone reaches its range band, turn-to-face, mark, and
weapon actions are pipelined on the same drone queue.
"""

from __future__ import annotations

import math
from collections.abc import Container
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
from agent.planning.tasks import Strike
from agent.rules.costs import estimate_battery_cost
from agent.rules.geometry import (
    absolute_to_relative,
    distance_3d,
    has_line_of_sight,
    hex_line,
)
from agent.rules.hexmath import get_neighbor, hex_distance_cube, hex_tiles_in_range
from agent.sim.entity_sim import DroneSimState, EntitySim
from agent.transport.action_tracker import Precondition
from agent.world.model import DroneRecord, EnemyBuildingRecord, WorldModel
from agent.world.tiles import Coord, Terrain
from agent.world.tracks import EnemyTrack, TrackStore


LASER_BASE_RANGE = 10
AUTO_CANNON_RANGES = {1: 5, 2: 10, 3: 15}
AUTO_CANNON_MIN_RANGE = 2
CANISTER_BASE_RANGE = 3
MARK_RANGES = {1: 5, 2: 7, 3: 10}


class FighterPhase(str, Enum):
    """Observable engagement state for one fighter plan."""

    WAITING = "waiting"
    ARMING = "arming"
    CLOSING = "closing"
    OPENING = "opening"
    REPOSITIONING = "repositioning"
    ENGAGING = "engaging"
    DISENGAGING = "disengaging"
    HOLDING = "holding"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class FighterConfig:
    """Range, fire-control, and survival policy."""

    preferred_range: int = 4
    range_tolerance: int = 1
    health_disengage_fraction: float = 0.30
    battery_disengage_fraction: float = 0.20
    minimum_track_confidence: float = 0.25
    move_level: int = 1
    mark_level: int = 1
    laser_level: int = 1
    auto_cannon_level: int = 1
    mark_before_fire: bool = True
    retreat_coord: Coord | None = None
    unknown_los_blocking: bool = True
    ammo_preference: tuple[str, ...] = (
        "armour_piercing_ammunition",
        "high_explosive_ammunition",
        "canister_round_ammunition",
        "ammunition",
    )

    def __post_init__(self) -> None:
        if self.preferred_range < 1:
            raise ValueError("preferred_range must be positive")
        if self.range_tolerance < 0:
            raise ValueError("range_tolerance cannot be negative")
        for name in (
            "health_disengage_fraction",
            "battery_disengage_fraction",
            "minimum_track_confidence",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("move_level", "mark_level", "laser_level", "auto_cannon_level"):
            if not 1 <= getattr(self, name) <= 3:
                raise ValueError(f"{name} must be between 1 and 3")
        if self.retreat_coord is not None and (
            not isinstance(self.retreat_coord, tuple)
            or len(self.retreat_coord) != 2
            or not all(isinstance(value, int) for value in self.retreat_coord)
        ):
            raise ValueError("retreat_coord must be an integer (q, r) coordinate")
        if not self.ammo_preference:
            raise ValueError("ammo_preference cannot be empty")


@dataclass(frozen=True, slots=True)
class EngagementTarget:
    """Resolved target used by one immutable fighter plan."""

    target_id: str
    coord: Coord
    kind: str
    confidence: float = 1.0
    marked: bool = False
    track_id: str | None = None


@dataclass(frozen=True, slots=True)
class FighterPlan:
    """The next queue-safe engagement leg."""

    drone_id: str
    phase: FighterPhase
    actions: tuple[PlannedAction, ...] = ()
    target: EngagementTarget | None = None
    destination: Coord | None = None
    preferred_min_range: int = 0
    preferred_max_range: int = 0
    reason: str = ""


@dataclass(frozen=True, slots=True)
class _WeaponPolicy:
    laser: bool
    auto_cannon: bool
    auto_loaded: bool
    load_ammo: str | None
    minimum_range: int
    maximum_range: int
    preferred_min: int
    preferred_max: int
    laser_level: int
    auto_level: int


@dataclass(frozen=True, slots=True)
class _Route:
    result: PathResult
    actions: tuple[PlannedAction, ...]
    destination: Coord
    end_heading: int
    battery_cost: int


class FighterController:
    """Plan range control and volleys for one fighter drone.

    ``marked_targets`` is a live shared collection of track ids, drone ids, or
    building ids.  Autonomous target selection ranks these first, which lets
    several controller instances focus fire without sharing mutable planner
    internals.  A concrete ``Strike`` target always remains authoritative.
    """

    def __init__(
        self,
        world: WorldModel,
        tracks: TrackStore,
        sim: EntitySim,
        drone_id: str,
        task: Strike | None = None,
        *,
        marked_targets: Container[str] = (),
        threat_cost: TileCost | None = None,
        comms_risk: TileCost | None = None,
        pathfinding: PathfindingConfig | None = None,
        movement: MovementProfile | None = None,
        allowed: AllowedTiles | None = None,
        config: FighterConfig | None = None,
    ) -> None:
        if not drone_id.strip():
            raise ValueError("drone_id cannot be empty")
        self._world = world
        self._tracks = tracks
        self._sim = sim
        self.drone_id = drone_id
        self.task = task
        self._marked_targets = marked_targets
        self._threat_cost = threat_cost
        self._comms_risk = comms_risk
        self._pathfinding = pathfinding
        self._movement = movement
        self._allowed = allowed
        self.config = config or FighterConfig()

    def plan(self) -> FighterPlan:
        """Return the next engagement leg from current authoritative state."""

        drone = self._world.get_drone(self.drone_id)
        state = self._sim.get_drone(self.drone_id)
        missing = self._missing_state(drone, state)
        if missing:
            return self._plan(FighterPhase.WAITING, reason=missing)
        assert drone is not None and drone.coord is not None and drone.direction is not None
        assert state is not None

        target = self._select_target(drone.coord)
        if self._should_disengage(state):
            return self._plan_disengage(drone, state, target)
        if target is None:
            return self._plan(
                FighterPhase.WAITING,
                reason="no fresh non-decoy target matches the engagement task",
            )
        if drone.elevation > 0:
            return self._plan(
                FighterPhase.BLOCKED,
                target=target,
                reason="fighter must descend to ground level before firing",
            )

        weapons = self._weapon_policy(state)
        if not weapons.laser and not weapons.auto_cannon:
            return self._plan(
                FighterPhase.BLOCKED,
                target=target,
                reason="fighter has no functional laser or auto-cannon with ammunition",
            )
        if weapons.auto_cannon and not weapons.auto_loaded:
            return self._plan_arm(state, target, weapons)

        distance = self._effective_distance(drone.coord, target.coord)
        if distance < weapons.preferred_min:
            return self._plan_range_move(FighterPhase.OPENING, drone, state, target, weapons)
        if distance > weapons.preferred_max:
            return self._plan_range_move(FighterPhase.CLOSING, drone, state, target, weapons)
        if not self._has_fire_position(drone.coord, target.coord, state, weapons):
            return self._plan_range_move(FighterPhase.REPOSITIONING, drone, state, target, weapons)
        return self._plan_volley(drone, state, target, weapons)

    def _plan_arm(
        self,
        state: DroneSimState,
        target: EngagementTarget,
        weapons: _WeaponPolicy,
    ) -> FighterPlan:
        if weapons.load_ammo is None:
            return self._plan(
                FighterPhase.BLOCKED,
                target=target,
                reason="auto-cannon has no available ammunition pack",
            )
        ammo_type = _ammo_payload_name(weapons.load_ammo)
        action = PlannedAction(
            "auto_cannon/load",
            {"ammo_type": ammo_type},
            precondition=self._arm_guard(weapons.load_ammo),
            distance=self._distance_from_base(),
        )
        return self._plan(
            FighterPhase.ARMING,
            actions=(action,),
            target=target,
            minimum=weapons.preferred_min,
            maximum=weapons.preferred_max,
            reason=f"load {ammo_type} before engaging",
        )

    def _plan_range_move(
        self,
        phase: FighterPhase,
        drone: DroneRecord,
        state: DroneSimState,
        target: EngagementTarget,
        weapons: _WeaponPolicy,
    ) -> FighterPlan:
        assert drone.coord is not None and drone.direction is not None
        finder = self._new_pathfinder()
        destinations = self._firing_positions(target.coord, state, weapons)
        route = self._route(
            finder,
            drone.coord,
            drone.direction,
            destinations,
            state,
            excluded=(target.coord,),
        )
        if route is None:
            return self._plan(
                FighterPhase.BLOCKED,
                target=target,
                minimum=weapons.preferred_min,
                maximum=weapons.preferred_max,
                reason="no reachable firing position in the preferred range band",
            )
        if not route.actions:
            return self._plan(
                FighterPhase.HOLDING,
                target=target,
                destination=route.destination,
                minimum=weapons.preferred_min,
                maximum=weapons.preferred_max,
                reason="already at the best currently reachable firing position",
            )
        if route.battery_cost >= state.current_battery:
            return self._plan(
                FighterPhase.BLOCKED,
                target=target,
                destination=route.destination,
                minimum=weapons.preferred_min,
                maximum=weapons.preferred_max,
                reason="range-control route would exhaust the fighter battery",
            )
        return self._plan(
            phase,
            actions=route.actions,
            target=target,
            destination=route.destination,
            minimum=weapons.preferred_min,
            maximum=weapons.preferred_max,
        )

    def _plan_volley(
        self,
        drone: DroneRecord,
        state: DroneSimState,
        target: EngagementTarget,
        weapons: _WeaponPolicy,
    ) -> FighterPlan:
        assert drone.coord is not None and drone.direction is not None
        facing = _direction_towards(drone.coord, target.coord)
        actions = list(self._turn_actions(drone.coord, drone.direction, facing))
        relative = absolute_to_relative(target.coord, origin=drone.coord)
        distance = self._effective_distance(drone.coord, target.coord)
        guard = self._attack_guard(target, drone.coord, facing, weapons)

        mark_range = MARK_RANGES[self.config.mark_level] + 3 * self._sim.research_tiers.get(
            "targeting_computer", 0
        )
        if (
            self.config.mark_before_fire
            and "targeting_computer" in state.functional_equipment
            and distance <= mark_range
        ):
            actions.append(
                PlannedAction(
                    "targeting_computer/mark",
                    {
                        "target_q": relative[0],
                        "target_r": relative[1],
                        "level": self.config.mark_level,
                    },
                    precondition=guard,
                    distance=self._distance_from_base(),
                )
            )

        if weapons.auto_cannon and weapons.auto_loaded and distance >= AUTO_CANNON_MIN_RANGE:
            auto_max = self._auto_range(state, weapons.auto_level)
            if distance <= auto_max and self._auto_has_los(drone.coord, target.coord, state):
                actions.append(
                    PlannedAction(
                        "auto_cannon/fire",
                        {
                            "target_q": relative[0],
                            "target_r": relative[1],
                            "level": weapons.auto_level,
                        },
                        precondition=guard,
                        distance=self._distance_from_base(),
                    )
                )
        if weapons.laser and distance <= self._laser_range():
            if self._laser_has_los(drone.coord, target.coord, drone):
                actions.append(
                    PlannedAction(
                        "laser_cannon/fire",
                        {
                            "target_q": relative[0],
                            "target_r": relative[1],
                            "level": weapons.laser_level,
                        },
                        precondition=guard,
                        distance=self._distance_from_base(),
                    )
                )

        weapon_actions = [action for action in actions if action.action.endswith("/fire")]
        if not weapon_actions:
            return self._plan(
                FighterPhase.REPOSITIONING,
                target=target,
                minimum=weapons.preferred_min,
                maximum=weapons.preferred_max,
                reason="target is in range but no functional weapon has a clear shot",
            )
        battery_cost = sum(self._battery_cost(action, state) for action in actions)
        floor = math.ceil(state.max_battery * self.config.battery_disengage_fraction)
        if state.current_battery - battery_cost < floor:
            return self._plan_disengage(drone, state, target)
        return self._plan(
            FighterPhase.ENGAGING,
            actions=tuple(actions),
            target=target,
            destination=drone.coord,
            minimum=weapons.preferred_min,
            maximum=weapons.preferred_max,
            reason="turn to face, mark when available, then fire all ready weapons",
        )

    def _plan_disengage(
        self,
        drone: DroneRecord,
        state: DroneSimState,
        target: EngagementTarget | None,
    ) -> FighterPlan:
        assert drone.coord is not None and drone.direction is not None
        finder = self._new_pathfinder()
        destinations = self._retreat_positions(drone.coord, target)
        route = self._route(
            finder,
            drone.coord,
            drone.direction,
            destinations,
            state,
            excluded=(target.coord,) if target is not None else (),
        )
        reason = self._disengage_reason(state)
        if route is None:
            return self._plan(
                FighterPhase.BLOCKED,
                target=target,
                reason=f"{reason}; no reachable retreat position",
            )
        if not route.actions:
            return self._plan(
                FighterPhase.HOLDING,
                target=target,
                destination=route.destination,
                reason=f"{reason}; holding at retreat position",
            )
        if route.battery_cost >= state.current_battery:
            return self._plan(
                FighterPhase.BLOCKED,
                target=target,
                destination=route.destination,
                reason=f"{reason}; retreat route would exhaust battery",
            )
        return self._plan(
            FighterPhase.DISENGAGING,
            actions=route.actions,
            target=target,
            destination=route.destination,
            reason=reason,
        )

    def _weapon_policy(self, state: DroneSimState) -> _WeaponPolicy:
        laser = "laser_cannon" in state.functional_equipment
        auto = "auto_cannon" in state.functional_equipment
        load_ammo = next(
            (ammo for ammo in self.config.ammo_preference if state.consumables.get(ammo, 0) > 0),
            None,
        )
        auto_loaded = bool(
            auto and state.loaded_ammo_type and state.consumables.get(state.loaded_ammo_type, 0) > 0
        )
        auto_usable = auto and (auto_loaded or load_ammo is not None)
        laser_level = state.equipment_levels.get("laser_cannon", self.config.laser_level)
        auto_level = state.equipment_levels.get("auto_cannon", self.config.auto_cannon_level)
        maximums: list[int] = []
        minimum = 0
        if laser:
            maximums.append(self._laser_range())
        if auto_usable:
            maximums.append(self._auto_range(state, auto_level))
            minimum = AUTO_CANNON_MIN_RANGE
        maximum = min(maximums) if maximums else 0
        preferred = min(maximum, max(minimum, self.config.preferred_range))
        preferred_min = max(minimum, preferred - self.config.range_tolerance)
        preferred_max = min(maximum, preferred + self.config.range_tolerance)
        return _WeaponPolicy(
            laser=laser,
            auto_cannon=auto_usable,
            auto_loaded=auto_loaded,
            load_ammo=load_ammo,
            minimum_range=minimum,
            maximum_range=maximum,
            preferred_min=preferred_min,
            preferred_max=preferred_max,
            laser_level=laser_level,
            auto_level=auto_level,
        )

    def _select_target(self, origin: Coord) -> EngagementTarget | None:
        if self.task is not None:
            if isinstance(self.task.target, tuple):
                return EngagementTarget(
                    target_id=f"coord:{self.task.target[0]},{self.task.target[1]}",
                    coord=self.task.target,
                    kind="coordinate",
                    marked=False,
                )
            return self._target_by_id(self.task.target)

        candidates: list[EngagementTarget] = []
        now = self._world.cycle
        for track in self._tracks.tracks():
            target = self._target_from_track(track, now)
            if target is not None:
                candidates.append(target)
        for building in self._world.enemy_buildings():
            target = self._target_from_building(building)
            if target is not None:
                candidates.append(target)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda target: (
                not target.marked,
                hex_distance_cube(*origin, *target.coord),
                -target.confidence,
                target.target_id,
            ),
        )

    def _target_by_id(self, target_id: str) -> EngagementTarget | None:
        now = self._world.cycle
        for track in self._tracks.tracks():
            if target_id in {track.track_id, track.drone_id}:
                return self._target_from_track(track, now)
        building = self._world.get_enemy_building(target_id)
        return self._target_from_building(building) if building is not None else None

    def _target_from_track(self, track: EnemyTrack, now_cycle: int) -> EngagementTarget | None:
        if track.is_decoy:
            return None
        cycle = max(now_cycle, track.last_seen)
        confidence = track.confidence(cycle)
        if confidence < self.config.minimum_track_confidence:
            return None
        identifiers = {track.track_id}
        if track.drone_id is not None:
            identifiers.add(track.drone_id)
        return EngagementTarget(
            target_id=track.drone_id or track.track_id,
            coord=track.predict(cycle),
            kind="drone",
            confidence=confidence,
            marked=any(identifier in self._marked_targets for identifier in identifiers),
            track_id=track.track_id,
        )

    def _target_from_building(self, building: EnemyBuildingRecord) -> EngagementTarget | None:
        if building.origin is None:
            return None
        return EngagementTarget(
            target_id=building.building_id,
            coord=building.origin,
            kind="building",
            marked=building.building_id in self._marked_targets,
        )

    def _firing_positions(
        self,
        target: Coord,
        state: DroneSimState,
        weapons: _WeaponPolicy,
    ) -> frozenset[Coord]:
        positions = {
            coord
            for coord in hex_tiles_in_range(target[0], target[1], weapons.preferred_max)
            if weapons.preferred_min
            <= self._effective_distance(coord, target)
            <= weapons.preferred_max
            and coord != target
            and self._has_fire_position(coord, target, state, weapons)
        }
        return frozenset(positions)

    def _has_fire_position(
        self,
        origin: Coord,
        target: Coord,
        state: DroneSimState,
        weapons: _WeaponPolicy,
    ) -> bool:
        distance = self._effective_distance(origin, target)
        if weapons.laser and distance <= self._laser_range():
            drone = self._world.get_drone(self.drone_id)
            if drone is not None and self._laser_has_los(origin, target, drone):
                return True
        if (
            weapons.auto_cannon
            and weapons.auto_loaded
            and AUTO_CANNON_MIN_RANGE <= distance <= self._auto_range(state, weapons.auto_level)
            and self._auto_has_los(origin, target, state)
        ):
            return True
        return False

    def _laser_has_los(self, origin: Coord, target: Coord, drone: DroneRecord) -> bool:
        if not self._clear_projectile_path(origin, target):
            return False
        elevations = {
            tile.coord: float(tile.elevation)
            for tile in self._world.tiles()
            if tile.elevation is not None
        }
        origin_tile = self._world.get_tile(origin)
        target_tile = self._world.get_tile(target)
        origin_terrain = origin_tile.elevation if origin_tile and origin_tile.elevation else 0
        target_terrain = target_tile.elevation if target_tile and target_tile.elevation else 0
        if target_terrain < origin_terrain + drone.elevation:
            return False
        return has_line_of_sight(
            origin,
            target,
            elevations,
            origin_elevation=drone.elevation,
            unknown_is_blocking=self.config.unknown_los_blocking,
        )

    def _auto_has_los(self, origin: Coord, target: Coord, state: DroneSimState) -> bool:
        if state.loaded_ammo_type == "canister_round_ammunition":
            return True
        drone = self._world.get_drone(self.drone_id)
        return drone is not None and self._laser_has_los(origin, target, drone)

    def _clear_projectile_path(self, origin: Coord, target: Coord) -> bool:
        for coord in hex_line(origin, target)[1:-1]:
            tile = self._world.get_tile(coord)
            if tile is None:
                if self.config.unknown_los_blocking:
                    return False
                continue
            if tile.terrain_type == Terrain.IMPASSABLE or tile.has_resource or tile.has_building:
                return False
        return True

    def _effective_distance(self, origin: Coord, target: Coord) -> float:
        drone = self._world.get_drone(self.drone_id)
        origin_tile = self._world.get_tile(origin)
        target_tile = self._world.get_tile(target)
        origin_height = float(origin_tile.elevation or 0) if origin_tile else 0.0
        if drone is not None:
            origin_height += drone.elevation
        target_height = float(target_tile.elevation or 0) if target_tile else 0.0
        return distance_3d(
            origin,
            target,
            origin_elevation=origin_height,
            target_elevation=target_height,
        )

    def _retreat_positions(
        self, origin: Coord, target: EngagementTarget | None
    ) -> frozenset[Coord]:
        if self.config.retreat_coord is not None:
            return frozenset((self.config.retreat_coord,))
        command_centers = [
            building
            for building in self._world.buildings()
            if building.building_type == "command_center"
        ]
        if command_centers:
            tiles = set(command_centers[0].tiles)
            if not tiles and command_centers[0].origin is not None:
                tiles.add(command_centers[0].origin)
            service = {
                get_neighbor(*tile, direction)
                for tile in tiles
                for direction in range(6)
                if get_neighbor(*tile, direction) not in tiles
            }
            if service:
                return frozenset(service)
        if target is None:
            return frozenset((origin,))
        current_distance = hex_distance_cube(*origin, *target.coord)
        desired = max(self.config.preferred_range + 2, current_distance + 2)
        return frozenset(
            coord
            for coord in hex_tiles_in_range(*target.coord, desired)
            if hex_distance_cube(*coord, *target.coord) == desired
        )

    def _route(
        self,
        finder: Pathfinder,
        start: Coord,
        heading: int,
        destinations: Container[Coord],
        state: DroneSimState,
        *,
        excluded: Container[Coord] = (),
    ) -> _Route | None:
        best: tuple[float, int, Coord, PathResult, tuple[PlannedAction, ...], int] | None = None
        for destination in sorted(destinations):
            result = finder.find_path(
                start,
                destination,
                allowed=_without_tiles(self._allowed, excluded),
            )
            if result is None:
                continue
            actions = result.compile(
                heading,
                level=self.config.move_level,
                precondition_factory=self._motion_guard,
            )
            battery = sum(self._battery_cost(action, state) for action in actions)
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
        distance = float(hex_distance_cube(0, 0, *best[2]))
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

    def _turn_actions(
        self, position: Coord, initial: int, target: int
    ) -> tuple[PlannedAction, ...]:
        clockwise = (target - initial) % 6
        direction = 1 if clockwise <= 3 else -1
        count = clockwise if clockwise <= 3 else 6 - clockwise
        heading = initial
        actions: list[PlannedAction] = []
        for _ in range(count):
            actions.append(
                PlannedAction(
                    "propulsion/turn",
                    {"direction": direction, "level": self.config.move_level},
                    precondition=self._motion_guard(position, heading),
                    distance=self._distance_from_base(),
                )
            )
            heading = (heading + direction) % 6
        return tuple(actions)

    def _motion_guard(self, position: Coord, heading: int) -> Precondition:
        def current() -> bool:
            drone = self._world.get_drone(self.drone_id)
            return drone is not None and drone.coord == position and drone.direction == heading

        return current

    def _attack_guard(
        self,
        target: EngagementTarget,
        position: Coord,
        heading: int,
        weapons: _WeaponPolicy,
    ) -> Precondition:
        def current() -> bool:
            drone = self._world.get_drone(self.drone_id)
            state = self._sim.get_drone(self.drone_id)
            current_target = self._target_by_id(target.target_id)
            if target.kind == "coordinate":
                current_target = target
            return (
                drone is not None
                and drone.coord == position
                and drone.direction == heading
                and state is not None
                and not self._should_disengage(state)
                and current_target is not None
                and current_target.coord == target.coord
                and (weapons.laser or weapons.auto_cannon)
            )

        return current

    def _arm_guard(self, ammo_type: str) -> Precondition:
        def current() -> bool:
            state = self._sim.get_drone(self.drone_id)
            return (
                state is not None
                and "auto_cannon" in state.functional_equipment
                and state.loaded_ammo_type is None
                and state.consumables.get(ammo_type, 0) > 0
                and not self._should_disengage(state)
            )

        return current

    def _should_disengage(self, state: DroneSimState) -> bool:
        return (
            state.health_fraction <= self.config.health_disengage_fraction
            or state.battery_fraction <= self.config.battery_disengage_fraction
        )

    def _disengage_reason(self, state: DroneSimState) -> str:
        thresholds: list[str] = []
        if state.health_fraction <= self.config.health_disengage_fraction:
            thresholds.append(f"health {state.health_fraction:.0%}")
        if state.battery_fraction <= self.config.battery_disengage_fraction:
            thresholds.append(f"battery {state.battery_fraction:.0%}")
        return "disengage threshold crossed: " + ", ".join(thresholds)

    def _laser_range(self) -> int:
        return LASER_BASE_RANGE + 2 * self._sim.research_tiers.get("laser_cannon", 0)

    def _auto_range(self, state: DroneSimState, level: int) -> int:
        if state.loaded_ammo_type == "canister_round_ammunition":
            return CANISTER_BASE_RANGE + self._sim.research_tiers.get("canister_round", 0)
        return AUTO_CANNON_RANGES[level] + 2 * self._sim.research_tiers.get("auto_cannon", 0)

    def _battery_cost(self, action: PlannedAction, state: DroneSimState) -> int:
        subject = {
            "auto_cannon/fire": "fire cannon completed",
            "auto_cannon/load": "autocannon load completed",
            "auto_cannon/unload": "autocannon unload completed",
        }.get(action.action, f"{action.action.replace('_', ' ')} completed")
        return estimate_battery_cost(
            subject,
            action.payload,
            research_tiers=self._sim.research_tiers,
            total_weight=state.total_weight,
        )

    def _new_pathfinder(self) -> Pathfinder:
        return Pathfinder(
            self._world,
            threat_cost=self._threat_cost,
            comms_risk=self._comms_risk,
            config=self._pathfinding,
            movement=self._movement,
        )

    def _distance_from_base(self) -> float:
        drone = self._world.get_drone(self.drone_id)
        return float(hex_distance_cube(0, 0, *drone.coord)) if drone and drone.coord else 0.0

    @staticmethod
    def _missing_state(drone: DroneRecord | None, state: DroneSimState | None) -> str:
        if drone is None or drone.coord is None:
            return "waiting for fighter position checkpoint"
        if drone.direction is None:
            return "waiting for fighter heading checkpoint"
        if state is None or state.max_battery <= 0:
            return "waiting for fighter battery checkpoint"
        if state.max_health <= 0:
            return "waiting for fighter health checkpoint"
        return ""

    def _plan(
        self,
        phase: FighterPhase,
        *,
        actions: tuple[PlannedAction, ...] = (),
        target: EngagementTarget | None = None,
        destination: Coord | None = None,
        minimum: int = 0,
        maximum: int = 0,
        reason: str = "",
    ) -> FighterPlan:
        return FighterPlan(
            drone_id=self.drone_id,
            phase=phase,
            actions=actions,
            target=target,
            destination=destination,
            preferred_min_range=minimum,
            preferred_max_range=maximum,
            reason=reason,
        )


def _direction_towards(origin: Coord, target: Coord) -> int:
    if origin == target:
        return 0
    return min(
        range(6),
        key=lambda direction: (
            hex_distance_cube(*get_neighbor(*origin, direction), *target),
            direction,
        ),
    )


def _heading_after(initial: int, actions: tuple[PlannedAction, ...]) -> int:
    heading = initial
    for action in actions:
        if action.action == "propulsion/turn":
            heading = (heading + int(action.payload.get("direction", 0))) % 6
    return heading


def _ammo_payload_name(canonical: str) -> str:
    return {
        "armour_piercing_ammunition": "armour_piercing",
        "high_explosive_ammunition": "high_explosive",
        "canister_round_ammunition": "canister_round",
        "ammunition": "armour_piercing",
    }[canonical]


def _without_tiles(allowed: AllowedTiles | None, excluded: Container[Coord]) -> AllowedTiles:
    def permits(coord: Coord) -> bool:
        if coord in excluded:
            return False
        if allowed is None:
            return True
        return bool(allowed(coord)) if callable(allowed) else coord in allowed

    return permits


__all__ = [
    "EngagementTarget",
    "FighterConfig",
    "FighterController",
    "FighterPhase",
    "FighterPlan",
]
