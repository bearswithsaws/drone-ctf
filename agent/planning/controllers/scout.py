"""Battery-safe frontier scouting for the live match loop.

Wire coordinates are relative to the acting unit. Ingest normalises those
observations into the command-centre-relative frame held by ``WorldModel``;
this controller plans only inside that shared local frame.

A scout is unarmed, so contact is always a reason to leave rather than to
trade: :meth:`ScoutController.plan` checks for hostiles before it looks for a
frontier, and withdraws towards the base while staying inside the same
known-tile discipline used by the exploration legs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from agent.planning.pathfind import Pathfinder, PathfindingConfig, TileCost
from agent.planning.pipeliner import PlannedAction
from agent.planning.tasks import ScoutSector
from agent.rules.costs import estimate_battery_cost
from agent.rules.hexmath import get_neighbor, hex_distance_cube, hex_tiles_in_range
from agent.sim.entity_sim import DroneSimState, EntitySim
from agent.transport.action_tracker import Precondition
from agent.world.model import DroneRecord, WorldModel
from agent.world.tiles import Coord, Terrain
from agent.world.tracks import TrackStore


class ScoutPhase(str, Enum):
    WAITING = "waiting"
    TRAVELLING = "travelling"
    SCANNING = "scanning"
    RETREATING = "retreating"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ScoutConfig:
    # Match the strategist's emergency-charge trigger so a scout leg never
    # deliberately crosses below the point where normal work is pre-empted.
    reserve_fraction: float = 0.30
    minimum_reserve: int = 100
    max_leg_tiles: int = 4
    max_threat_cost: float = 0.0
    # Retreat policy.  ``contact_radius`` is deliberately wider than a scout's
    # own scan so a hostile that appears at the edge of the sensor picture is
    # already treated as contact; ``max_retreat_tiles`` bounds one withdrawal
    # leg, and replanning continues the withdrawal while the contact holds.
    contact_radius: int = 8
    minimum_contact_confidence: float = 0.25
    max_retreat_tiles: int = 6
    retreat_coord: Coord | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.reserve_fraction < 1:
            raise ValueError("reserve_fraction must be in [0, 1)")
        if self.minimum_reserve < 0:
            raise ValueError("minimum_reserve must be non-negative")
        if self.max_leg_tiles < 1:
            raise ValueError("max_leg_tiles must be positive")
        if not math.isfinite(self.max_threat_cost) or self.max_threat_cost < 0:
            raise ValueError("max_threat_cost must be finite and non-negative")
        if self.contact_radius < 0:
            raise ValueError("contact_radius must be non-negative")
        if not 0 <= self.minimum_contact_confidence <= 1:
            raise ValueError("minimum_contact_confidence must be in [0, 1]")
        if self.max_retreat_tiles < 1:
            raise ValueError("max_retreat_tiles must be positive")
        if self.retreat_coord is not None and (
            not isinstance(self.retreat_coord, tuple)
            or len(self.retreat_coord) != 2
            or not all(isinstance(value, int) for value in self.retreat_coord)
        ):
            raise ValueError("retreat_coord must be an integer (q, r) coordinate")


@dataclass(frozen=True, slots=True)
class ScoutContact:
    """One hostile the scout currently believes is close enough to matter."""

    contact_id: str
    coord: Coord
    distance: int
    confidence: float = 1.0
    drone_id: str | None = None


@dataclass(frozen=True, slots=True)
class ScoutPlan:
    drone_id: str
    phase: ScoutPhase
    actions: tuple[PlannedAction, ...] = ()
    destination: Coord | None = None
    reason: str = ""
    contacts: tuple[ScoutContact, ...] = ()


class ScoutController:
    """Advance over known-safe tiles to a frontier, then scan from there.

    ``tracks`` is optional only so crafted maps and unit tests can plan without
    a track store; the live wiring always supplies one, because enemy drones
    are the contacts a scout most needs to run from.
    """

    def __init__(
        self,
        world: WorldModel,
        sim: EntitySim,
        drone_id: str,
        task: ScoutSector,
        *,
        tracks: TrackStore | None = None,
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
        self._tracks = tracks
        self._threat_cost = threat_cost
        self._pathfinding = pathfinding
        self.config = config or ScoutConfig()

    def contacts(self) -> tuple[ScoutContact, ...]:
        """Return the hostiles inside the contact radius, nearest first.

        Exposed so a live output can pre-empt a queued exploration leg the
        moment a contact appears, without waiting for the leg to drain.
        """

        drone = self._world.get_drone(self.drone_id)
        if drone is None or drone.coord is None or self._tracks is None:
            return ()
        now = self._world.cycle
        found: list[ScoutContact] = []
        for track in self._tracks.tracks():
            if track.is_decoy:
                continue
            cycle = max(now, track.last_seen)
            confidence = track.confidence(cycle)
            if confidence < self.config.minimum_contact_confidence:
                continue
            coord = track.predict(cycle)
            distance = hex_distance_cube(*drone.coord, *coord)
            if distance > self.config.contact_radius:
                continue
            found.append(
                ScoutContact(
                    contact_id=track.track_id,
                    coord=coord,
                    distance=distance,
                    confidence=confidence,
                    drone_id=track.drone_id,
                )
            )
        return tuple(sorted(found, key=lambda c: (c.distance, -c.confidence, c.contact_id)))

    def plan(self) -> ScoutPlan:
        drone = self._world.get_drone(self.drone_id)
        missing = self._missing_state(drone)
        if missing:
            return ScoutPlan(self.drone_id, ScoutPhase.WAITING, reason=missing)
        state = self._sim.get_drone(self.drone_id)
        assert drone is not None and drone.coord is not None and drone.direction is not None
        assert state is not None

        contacts = self.contacts()
        exposed = self._threat_at(drone.coord) > self.config.max_threat_cost
        if contacts or exposed:
            return self._plan_retreat(drone, state, contacts, exposed)

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
            return ScoutPlan(
                self.drone_id,
                ScoutPhase.TRAVELLING if movement else ScoutPhase.SCANNING,
                self._stamped(actions, destination),
                destination,
            )

        return ScoutPlan(
            self.drone_id,
            ScoutPhase.BLOCKED,
            reason="frontier leg would violate battery reserve",
        )

    def _plan_retreat(
        self,
        drone: DroneRecord,
        state: DroneSimState,
        contacts: tuple[ScoutContact, ...],
        exposed: bool,
    ) -> ScoutPlan:
        """Withdraw one leg towards safety; never scan while breaking contact."""

        assert drone.coord is not None and drone.direction is not None
        reason = self._retreat_reason(contacts, exposed)
        # Threatened tiles stay usable here: a scout that is already inside a
        # weapon envelope has to cross it to get out.  The pathfinder's threat
        # weight still steers the route towards the quietest way home.
        allowed = self._enterable_tiles(drone.coord, threat_limit=None)
        candidates = self._retreat_candidates(drone.coord, contacts, allowed)
        if not candidates:
            return ScoutPlan(
                self.drone_id,
                ScoutPhase.BLOCKED,
                destination=drone.coord,
                reason=f"{reason}; no safer tile within one retreat leg",
                contacts=contacts,
            )

        finder = Pathfinder(
            self._world,
            threat_cost=self._threat_cost,
            config=self._pathfinding,
        )
        for destination in candidates:
            route = finder.find_path(drone.coord, destination, allowed=allowed)
            if route is None or len(route.path) - 1 > self.config.max_retreat_tiles:
                continue
            actions = route.compile(
                drone.direction,
                level=self._equipment_level("propulsion"),
                precondition_factory=self._motion_guard,
            )
            # Breaking contact outranks the scouting reserve, but a leg that
            # would flatten the battery mid-route strands the drone in the open.
            if not actions or self._battery_cost(state, actions) >= state.current_battery:
                continue
            return ScoutPlan(
                self.drone_id,
                ScoutPhase.RETREATING,
                self._stamped(actions, destination),
                destination,
                reason,
                contacts,
            )

        return ScoutPlan(
            self.drone_id,
            ScoutPhase.BLOCKED,
            destination=drone.coord,
            reason=f"{reason}; no affordable retreat leg",
            contacts=contacts,
        )

    def _retreat_candidates(
        self,
        current: Coord,
        contacts: tuple[ScoutContact, ...],
        allowed: set[Coord],
    ) -> list[Coord]:
        """Rank enterable tiles that are strictly safer than the current one."""

        rally = self._rally_point()
        # Rank once per tile: a threat provider can be an expensive live query.
        ranks = {
            coord: self._retreat_rank(coord, contacts, rally)
            for coord in hex_tiles_in_range(*current, self.config.max_retreat_tiles)
            if coord in allowed
        }
        here = ranks[current]
        candidates = [
            coord for coord, rank in ranks.items() if coord != current and rank[:2] < here[:2]
        ]
        return sorted(candidates, key=lambda coord: ranks[coord])

    def _retreat_rank(
        self, coord: Coord, contacts: tuple[ScoutContact, ...], rally: Coord
    ) -> tuple[float, int, int, int, int]:
        # Leaving weapon envelopes first, then opening the range on the nearest
        # contact, then closing on the rally point.
        nearest = min(
            (hex_distance_cube(*coord, *contact.coord) for contact in contacts),
            default=0,
        )
        return (
            self._threat_at(coord),
            -nearest,
            hex_distance_cube(*coord, *rally),
            coord[0],
            coord[1],
        )

    def _rally_point(self) -> Coord:
        if self.config.retreat_coord is not None:
            return self.config.retreat_coord
        for building in self._world.buildings():
            if building.building_type == "command_center" and building.origin is not None:
                return building.origin
        # World coordinates are command-centre relative, so home is the origin.
        return (0, 0)

    @staticmethod
    def _retreat_reason(contacts: tuple[ScoutContact, ...], exposed: bool) -> str:
        if not contacts:
            return "scout is inside a hostile weapon envelope"
        nearest = contacts[0]
        label = nearest.drone_id or nearest.contact_id
        reason = f"hostile contact {label} at range {nearest.distance}"
        if len(contacts) > 1:
            reason += f" (+{len(contacts) - 1} more)"
        return f"{reason}; scout is inside a hostile weapon envelope" if exposed else reason

    def _known_safe_tiles(self, current: Coord) -> set[Coord]:
        return self._enterable_tiles(current, threat_limit=self.config.max_threat_cost)

    def _enterable_tiles(self, current: Coord, *, threat_limit: float | None) -> set[Coord]:
        occupied = {
            drone.coord
            for drone in self._world.drones()
            if drone.drone_id != self.drone_id and drone.coord is not None
        }
        tiles: set[Coord] = set()
        for tile in self._world.tiles():
            if tile.terrain_type not in (Terrain.NORMAL, Terrain.DIFFICULT):
                continue
            if tile.has_building or tile.coord in occupied:
                continue
            if tile.has_drone and tile.coord != current:
                continue
            if threat_limit is not None and self._threat_at(tile.coord) > threat_limit:
                continue
            tiles.add(tile.coord)
        # An initial status can arrive before a terrain scan. The tile under
        # the drone is necessarily enterable because the drone occupies it.
        tiles.add(current)
        return tiles

    @staticmethod
    def _stamped(
        actions: tuple[PlannedAction, ...], destination: Coord
    ) -> tuple[PlannedAction, ...]:
        """Re-stamp a leg with the range its actions are transmitted over."""

        distance = float(hex_distance_cube(0, 0, *destination))
        return tuple(
            PlannedAction(
                action.action,
                action.payload,
                precondition=action.precondition,
                distance=distance,
                endpoint=action.endpoint,
            )
            for action in actions
        )

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

    def _battery_cost(
        self, state: DroneSimState, actions: tuple[PlannedAction, ...]
    ) -> int:
        subjects = {
            "propulsion/drive": "Drive completed",
            "propulsion/turn": "Turn completed",
            "sensors/scan": "Scan completed",
        }
        return sum(
            estimate_battery_cost(
                subjects.get(action.action, action.action),
                action.payload,
                research_tiers=self._sim.research_tiers,
                total_weight=state.total_weight,
            )
            for action in actions
        )

    def _preserves_reserve(
        self, state: DroneSimState, actions: tuple[PlannedAction, ...]
    ) -> bool:
        reserve = max(
            self.config.minimum_reserve,
            math.ceil(state.max_battery * self.config.reserve_fraction),
        )
        return state.current_battery - self._battery_cost(state, actions) >= reserve

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


__all__ = [
    "ScoutConfig",
    "ScoutContact",
    "ScoutController",
    "ScoutPhase",
    "ScoutPlan",
]
