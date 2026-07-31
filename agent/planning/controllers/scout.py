"""Small, replanning-oriented frontier scout used by the live match loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agent.planning.pathfind import Pathfinder, PathfindingConfig, TileCost
from agent.planning.pipeliner import PlannedAction
from agent.planning.tasks import ScoutSector
from agent.rules.hexmath import hex_distance_cube, hex_tiles_in_range
from agent.sim.entity_sim import EntitySim
from agent.transport.action_tracker import Precondition
from agent.world.model import DroneRecord, WorldModel
from agent.world.tiles import Coord


class ScoutPhase(str, Enum):
    WAITING = "waiting"
    TRAVELLING = "travelling"
    SCANNING = "scanning"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ScoutPlan:
    drone_id: str
    phase: ScoutPhase
    actions: tuple[PlannedAction, ...] = ()
    destination: Coord | None = None
    reason: str = ""


class ScoutController:
    """Move toward the stalest tile in a sector, then refresh it with a scan."""

    def __init__(
        self,
        world: WorldModel,
        sim: EntitySim,
        drone_id: str,
        task: ScoutSector,
        *,
        threat_cost: TileCost | None = None,
        pathfinding: PathfindingConfig | None = None,
    ) -> None:
        if not drone_id.strip():
            raise ValueError("drone_id cannot be empty")
        self._world = world
        self._sim = sim
        self.drone_id = drone_id
        self.task = task
        self._threat_cost = threat_cost
        self._pathfinding = pathfinding

    def plan(self) -> ScoutPlan:
        drone = self._world.get_drone(self.drone_id)
        missing = self._missing_state(drone)
        if missing:
            return ScoutPlan(self.drone_id, ScoutPhase.WAITING, reason=missing)
        assert drone is not None and drone.coord is not None and drone.direction is not None

        destination = self._destination(drone.coord)
        finder = Pathfinder(
            self._world,
            threat_cost=self._threat_cost,
            config=self._pathfinding,
        )
        route = finder.find_path(drone.coord, destination)
        if route is None:
            return ScoutPlan(
                self.drone_id,
                ScoutPhase.BLOCKED,
                destination=destination,
                reason="frontier destination is unreachable",
            )
        actions = route.compile(
            drone.direction,
            level=self._equipment_level("propulsion"),
            precondition_factory=self._motion_guard,
        )
        if actions:
            distance = float(hex_distance_cube(0, 0, *destination))
            actions = tuple(
                PlannedAction(
                    action.action,
                    action.payload,
                    precondition=action.precondition,
                    distance=distance,
                )
                for action in actions
            )
            return ScoutPlan(
                self.drone_id,
                ScoutPhase.TRAVELLING,
                actions,
                destination,
            )

        scan = PlannedAction(
            "sensors/scan",
            {"level": self._equipment_level("sensors")},
            precondition=self._scan_guard(destination),
            distance=float(hex_distance_cube(0, 0, *destination)),
        )
        return ScoutPlan(
            self.drone_id,
            ScoutPhase.SCANNING,
            (scan,),
            destination,
        )

    def _destination(self, current: Coord) -> Coord:
        candidates = tuple(hex_tiles_in_range(*self.task.center, self.task.radius))

        def rank(coord: Coord) -> tuple[float, int, int, int]:
            # Unknown tiles have confidence zero.  Among equally stale tiles,
            # prefer a short leg and deterministic coordinates.
            return (
                self._world.confidence(coord),
                hex_distance_cube(*current, *coord),
                coord[0],
                coord[1],
            )

        return min(candidates, key=rank) if candidates else self.task.center

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
            return "waiting for drone equipment checkpoint"
        if "sensors" not in state.functional_equipment:
            return "drone has no functional sensors"
        if "propulsion" not in state.functional_equipment:
            return "drone has no functional propulsion"
        return ""


__all__ = ["ScoutController", "ScoutPhase", "ScoutPlan"]
