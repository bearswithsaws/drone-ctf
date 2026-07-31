"""Threat-aware A* pathfinding over the world model's tile beliefs.

The search is deliberately synchronous and side-effect free.  Callers running
large searches from the asyncio supervisor can put :meth:`Pathfinder.find_path`
in an executor without changing the planner API.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Container, Iterable
from dataclasses import dataclass
from itertools import count

from agent.planning.pipeliner import PlannedAction
from agent.rules.hexmath import get_neighbor, hex_distance_cube
from agent.transport.action_tracker import Precondition
from agent.world.model import WorldModel
from agent.world.tiles import Coord, Terrain, TileBelief

TileCost = Callable[[Coord], float]
AllowedTiles = Container[Coord] | Callable[[Coord], bool]
MotionPreconditionFactory = Callable[[Coord, int], Precondition]

DEFAULT_MOVE_COST = 1.0
DEFAULT_THREAT_WEIGHT = 10.0
DEFAULT_COMMS_WEIGHT = 1.0
DEFAULT_UNKNOWN_PENALTY = 2.0
DEFAULT_MAX_EXPANSIONS = 100_000


def _zero_cost(_coord: Coord) -> float:
    return 0.0


async def _never_retry() -> bool:
    return False


@dataclass(frozen=True, slots=True)
class MovementProfile:
    """Traversal capabilities that affect terrain and occupied tiles.

    Ground movement is the conservative default.  Callers may enable building
    overflight only after establishing that the drone's elevation clears the
    known footprint; unknown building clearance remains the caller's concern.
    """

    ignores_difficult_terrain: bool = False
    can_overfly_buildings: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.ignores_difficult_terrain, bool):
            raise ValueError("ignores_difficult_terrain must be a boolean")
        if not isinstance(self.can_overfly_buildings, bool):
            raise ValueError("can_overfly_buildings must be a boolean")


@dataclass(frozen=True, slots=True)
class PathfindingConfig:
    """Weights and safety limits used by a path search.

    Threat and communications providers return dimensionless costs.  Their
    weights convert those values into movement-cost equivalents.  Entering a
    difficult tile costs twice ``move_cost``; entering a tile whose terrain is
    not known also adds ``unknown_penalty``.
    """

    move_cost: float = DEFAULT_MOVE_COST
    threat_weight: float = DEFAULT_THREAT_WEIGHT
    comms_weight: float = DEFAULT_COMMS_WEIGHT
    unknown_penalty: float = DEFAULT_UNKNOWN_PENALTY
    max_expansions: int = DEFAULT_MAX_EXPANSIONS

    def __post_init__(self) -> None:
        for name in ("move_cost", "threat_weight", "comms_weight", "unknown_penalty"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.move_cost <= 0:
            raise ValueError("move_cost must be positive")
        if self.max_expansions < 1:
            raise ValueError("max_expansions must be positive")


@dataclass(frozen=True, slots=True)
class PathResult:
    """An optimal coordinate path and its weighted traversal cost."""

    path: tuple[Coord, ...]
    cost: float
    expanded: int

    @property
    def coordinates(self) -> tuple[Coord, ...]:
        """Descriptive alias for callers that avoid the ``result.path`` stutter."""

        return self.path

    def compile(
        self,
        initial_heading: int,
        *,
        level: int = 1,
        precondition_factory: MotionPreconditionFactory | None = None,
    ) -> tuple[PlannedAction, ...]:
        """Compile this route into actions accepted by :class:`Pipeliner`."""

        return compile_path(
            self.path,
            initial_heading,
            level=level,
            precondition_factory=precondition_factory,
        )


class SearchLimitExceeded(RuntimeError):
    """Raised when a search reaches its configured expansion safety limit."""


class Pathfinder:
    """A* over known and unknown odd-q hex tiles.

    Construction captures a detached tile snapshot and snapshots cost providers
    that expose a ``snapshot()`` method (including :class:`ThreatMap`).  Construct
    on the event-loop thread, then ``find_path`` can safely run in an executor.

    ``threat_cost`` is normally a :class:`agent.world.threat.ThreatMap`.
    ``comms_risk`` is injectable because the empirical E4 loss model is learned
    at runtime.  ``allowed`` can bound crafted maps or enforce mission-specific
    operating areas; without it, unknown tiles remain traversable at a penalty.
    """

    def __init__(
        self,
        world: WorldModel,
        *,
        threat_cost: TileCost | None = None,
        comms_risk: TileCost | None = None,
        config: PathfindingConfig | None = None,
        movement: MovementProfile | None = None,
    ) -> None:
        state = world.snapshot_state()
        self._tiles = {tile.coord: tile for tile in state.tiles}
        self._threat_cost = _snapshot_cost(threat_cost)
        self._comms_risk = _snapshot_cost(comms_risk)
        self.config = config or PathfindingConfig()
        self.movement = movement or MovementProfile()

    def find_path(
        self,
        start: Coord,
        goal: Coord,
        *,
        allowed: AllowedTiles | None = None,
    ) -> PathResult | None:
        """Return the least-cost path from ``start`` to ``goal``.

        The starting tile is included in the result and has no entry cost.
        Impassable goals and exhausted finite operating areas return ``None``.
        A search of the otherwise unbounded unknown world raises
        :class:`SearchLimitExceeded` at the configured safety limit.
        """

        _validate_coord(start, "start")
        _validate_coord(goal, "goal")
        if not _is_allowed(start, allowed) or not _is_allowed(goal, allowed):
            return None
        goal_tile = self._tiles.get(goal)
        if not self._can_enter(goal_tile):
            return None
        if start == goal:
            return PathResult((start,), 0.0, 0)

        frontier: list[tuple[float, int, float, Coord]] = []
        tie_breaker = count()
        heapq.heappush(
            frontier,
            (self._heuristic(start, goal), next(tie_breaker), 0.0, start),
        )
        best_cost: dict[Coord, float] = {start: 0.0}
        came_from: dict[Coord, Coord] = {}
        expanded = 0

        while frontier:
            _estimate, _order, current_cost, current = heapq.heappop(frontier)
            if current_cost > best_cost[current]:
                continue
            if current == goal:
                return PathResult(
                    _reconstruct_path(came_from, start, goal), current_cost, expanded
                )

            if expanded >= self.config.max_expansions:
                raise SearchLimitExceeded(
                    f"path search exceeded {self.config.max_expansions} expansions"
                )
            expanded += 1

            for direction in range(6):
                neighbor = get_neighbor(*current, direction)
                if not _is_allowed(neighbor, allowed):
                    continue
                tile = self._tiles.get(neighbor)
                if not self._can_enter(tile):
                    continue

                candidate_cost = current_cost + self._entry_cost(neighbor, tile)
                if candidate_cost >= best_cost.get(neighbor, math.inf):
                    continue
                best_cost[neighbor] = candidate_cost
                came_from[neighbor] = current
                estimate = candidate_cost + self._heuristic(neighbor, goal)
                heapq.heappush(
                    frontier,
                    (estimate, next(tie_breaker), candidate_cost, neighbor),
                )

        return None

    def _heuristic(self, coord: Coord, goal: Coord) -> float:
        return self.config.move_cost * hex_distance_cube(*coord, *goal)

    def _entry_cost(self, coord: Coord, tile: TileBelief | None) -> float:
        movement = self.config.move_cost
        unknown = tile is None or tile.terrain_type is None
        if (
            tile is not None
            and tile.terrain_type == Terrain.DIFFICULT
            and not self.movement.ignores_difficult_terrain
        ):
            movement *= 2

        threat = _checked_cost(self._threat_cost(coord), "threat_cost", coord)
        comms = _checked_cost(self._comms_risk(coord), "comms_risk", coord)
        return (
            movement
            + self.config.threat_weight * threat
            + self.config.comms_weight * comms
            + (self.config.unknown_penalty if unknown else 0.0)
        )

    def _can_enter(self, tile: TileBelief | None) -> bool:
        if tile is None:
            return True
        if not tile.passable:
            return False
        return not tile.has_building or self.movement.can_overfly_buildings


def find_path(
    world: WorldModel,
    start: Coord,
    goal: Coord,
    *,
    threat_cost: TileCost | None = None,
    comms_risk: TileCost | None = None,
    config: PathfindingConfig | None = None,
    movement: MovementProfile | None = None,
    allowed: AllowedTiles | None = None,
) -> PathResult | None:
    """Convenience wrapper for one A* search."""

    return Pathfinder(
        world,
        threat_cost=threat_cost,
        comms_risk=comms_risk,
        config=config,
        movement=movement,
    ).find_path(start, goal, allowed=allowed)


def compile_path(
    path: Iterable[Coord],
    initial_heading: int,
    *,
    level: int = 1,
    precondition_factory: MotionPreconditionFactory | None = None,
) -> tuple[PlannedAction, ...]:
    """Compile adjacent path coordinates into clockwise turns and forward drives.

    Each propulsion turn changes heading by one hex-side.  The shortest turn
    sequence is emitted before every forward drive; a 180-degree tie is made
    deterministic by choosing clockwise turns.  Motion retries fail closed by
    default: a controller that can verify expected position and heading may
    provide ``precondition_factory`` to safely recover genuinely lost commands.
    """

    coordinates = tuple(path)
    if not coordinates:
        raise ValueError("path cannot be empty")
    for index, coord in enumerate(coordinates):
        _validate_coord(coord, f"path[{index}]")
    if isinstance(initial_heading, bool) or not isinstance(initial_heading, int):
        raise ValueError("initial_heading must be between 0 and 5")
    if not 0 <= initial_heading <= 5:
        raise ValueError("initial_heading must be between 0 and 5")
    if isinstance(level, bool) or not isinstance(level, int):
        raise ValueError("level must be between 1 and 3")
    if not 1 <= level <= 3:
        raise ValueError("level must be between 1 and 3")

    heading = initial_heading
    actions: list[PlannedAction] = []
    for current, neighbor in zip(coordinates, coordinates[1:]):
        target_heading = _neighbor_direction(current, neighbor)
        clockwise = (target_heading - heading) % 6
        turn_direction = 1 if clockwise <= 3 else -1
        turn_count = clockwise if clockwise <= 3 else 6 - clockwise
        for _ in range(turn_count):
            precondition = _motion_precondition(precondition_factory, current, heading)
            actions.append(
                PlannedAction(
                    "propulsion/turn",
                    {"direction": turn_direction, "level": level},
                    precondition=precondition,
                )
            )
            heading = (heading + turn_direction) % 6
        precondition = _motion_precondition(precondition_factory, current, heading)
        actions.append(
            PlannedAction(
                "propulsion/drive",
                {"direction": 1, "level": level},
                precondition=precondition,
            )
        )
    return tuple(actions)


def _neighbor_direction(current: Coord, neighbor: Coord) -> int:
    for direction in range(6):
        if get_neighbor(*current, direction) == neighbor:
            return direction
    raise ValueError(f"path contains non-adjacent step: {current!r} -> {neighbor!r}")


def _reconstruct_path(
    came_from: dict[Coord, Coord], start: Coord, goal: Coord
) -> tuple[Coord, ...]:
    current = goal
    reversed_path = [current]
    while current != start:
        current = came_from[current]
        reversed_path.append(current)
    reversed_path.reverse()
    return tuple(reversed_path)


def _is_allowed(coord: Coord, allowed: AllowedTiles | None) -> bool:
    if allowed is None:
        return True
    if callable(allowed):
        return bool(allowed(coord))
    return coord in allowed


def _snapshot_cost(provider: TileCost | None) -> TileCost:
    if provider is None:
        return _zero_cost
    snapshot = getattr(provider, "snapshot", None)
    frozen = snapshot() if callable(snapshot) else provider
    if not callable(frozen):
        raise TypeError("tile-cost snapshot must be callable")
    return frozen


def _motion_precondition(
    factory: MotionPreconditionFactory | None,
    position: Coord,
    heading: int,
) -> Precondition:
    if factory is None:
        return _never_retry
    precondition = factory(position, heading)
    if not callable(precondition):
        raise TypeError("precondition_factory must return a callable")
    return precondition


def _checked_cost(value: float, provider: str, coord: Coord) -> float:
    cost = float(value)
    if not math.isfinite(cost) or cost < 0:
        raise ValueError(f"{provider} returned invalid cost {value!r} for {coord!r}")
    return cost


def _validate_coord(coord: Coord, name: str) -> None:
    if (
        not isinstance(coord, tuple)
        or len(coord) != 2
        or isinstance(coord[0], bool)
        or isinstance(coord[1], bool)
        or not isinstance(coord[0], int)
        or not isinstance(coord[1], int)
    ):
        raise ValueError(f"{name} must be a pair of integers")


__all__ = [
    "DEFAULT_COMMS_WEIGHT",
    "DEFAULT_MAX_EXPANSIONS",
    "DEFAULT_MOVE_COST",
    "DEFAULT_THREAT_WEIGHT",
    "DEFAULT_UNKNOWN_PENALTY",
    "MovementProfile",
    "MotionPreconditionFactory",
    "PathResult",
    "Pathfinder",
    "PathfindingConfig",
    "SearchLimitExceeded",
    "compile_path",
    "find_path",
]
