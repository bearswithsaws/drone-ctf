"""Typed strategic work items consumed by planning controllers.

Tasks describe *what* should be done, not how to do it.  The strategist can
therefore create them without knowing which entity will win the allocator's
auction, while controllers remain free to translate an assigned task into a
stream of :class:`~agent.planning.pipeliner.PlannedAction` values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, TypeAlias

from agent.world.tiles import Coord


TaskId: TypeAlias = str


def _non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} cannot be empty")


def _coord(value: Coord, field_name: str) -> Coord:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or not all(
            isinstance(component, int) and not isinstance(component, bool)
            for component in value
        )
    ):
        raise ValueError(f"{field_name} must be an integer (q, r) coordinate")
    return tuple(value)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Task:
    """Base task shared by drone and building work.

    ``priority`` is a multiplier applied to allocator fitness.  ``max_assignees``
    is an auction capacity; explicit commander pins may exceed it because an
    override must never be silently ignored.
    """

    task_id: TaskId
    priority: float = field(default=1.0, kw_only=True)
    max_assignees: int = field(default=1, kw_only=True)
    metadata: Mapping[str, Any] = field(
        default_factory=dict, kw_only=True, repr=False, compare=False, hash=False
    )
    kind: ClassVar[str] = "task"

    def __post_init__(self) -> None:
        _non_empty(self.task_id, "task_id")
        if (
            isinstance(self.priority, bool)
            or not isinstance(self.priority, (int, float))
            or not math.isfinite(self.priority)
            or self.priority <= 0
        ):
            raise ValueError("priority must be finite and positive")
        if isinstance(self.max_assignees, bool) or self.max_assignees < 1:
            raise ValueError("max_assignees must be a positive integer")
        if not isinstance(self.max_assignees, int):
            raise ValueError("max_assignees must be a positive integer")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class MineLoop(Task):
    """Continuously mine ``resource_node`` and unload at ``dropoff_id``."""

    resource_node: Coord
    dropoff_id: str
    resource_type: str | None = None
    kind: ClassVar[str] = "mine_loop"

    def __post_init__(self) -> None:
        Task.__post_init__(self)
        object.__setattr__(self, "resource_node", _coord(self.resource_node, "resource_node"))
        _non_empty(self.dropoff_id, "dropoff_id")
        if self.resource_type is not None:
            _non_empty(self.resource_type, "resource_type")


@dataclass(frozen=True, slots=True)
class ScoutSector(Task):
    """Explore and refresh observations around a sector center."""

    center: Coord
    radius: int = 1
    kind: ClassVar[str] = "scout_sector"

    def __post_init__(self) -> None:
        Task.__post_init__(self)
        object.__setattr__(self, "center", _coord(self.center, "center"))
        if isinstance(self.radius, bool) or not isinstance(self.radius, int) or self.radius < 0:
            raise ValueError("radius must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class Recharge(Task):
    """Safety override that keeps one drone charging to a recovery target."""

    drone_id: str
    target_fraction: float = 0.9
    kind: ClassVar[str] = "recharge"

    def __post_init__(self) -> None:
        Task.__post_init__(self)
        _non_empty(self.drone_id, "drone_id")
        if (
            isinstance(self.target_fraction, bool)
            or not isinstance(self.target_fraction, (int, float))
            or not math.isfinite(self.target_fraction)
            or not 0 < self.target_fraction <= 1
        ):
            raise ValueError("target_fraction must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class Escort(Task):
    """Protect another friendly entity."""

    target_id: str
    kind: ClassVar[str] = "escort"

    def __post_init__(self) -> None:
        Task.__post_init__(self)
        _non_empty(self.target_id, "target_id")


@dataclass(frozen=True, slots=True)
class Strike(Task):
    """Attack a known entity or coordinate."""

    target: str | Coord
    kind: ClassVar[str] = "strike"

    def __post_init__(self) -> None:
        Task.__post_init__(self)
        if isinstance(self.target, str):
            _non_empty(self.target, "target")
        else:
            object.__setattr__(self, "target", _coord(self.target, "target"))


@dataclass(frozen=True, slots=True)
class LayMines(Task):
    """Deploy mines at one or more absolute map coordinates."""

    positions: tuple[Coord, ...]
    kind: ClassVar[str] = "lay_mines"

    def __post_init__(self) -> None:
        Task.__post_init__(self)
        positions = tuple(_coord(position, "positions") for position in self.positions)
        if not positions:
            raise ValueError("positions cannot be empty")
        object.__setattr__(self, "positions", positions)


@dataclass(frozen=True, slots=True)
class DeployRepeater(Task):
    """Place a communications repeater at an absolute coordinate."""

    position: Coord
    kind: ClassVar[str] = "deploy_repeater"

    def __post_init__(self) -> None:
        Task.__post_init__(self)
        object.__setattr__(self, "position", _coord(self.position, "position"))


@dataclass(frozen=True, slots=True)
class Research(Task):
    """Run one research unlock or upgrade on a capable building."""

    research_key: str
    target_level: int | None = None
    kind: ClassVar[str] = "research"

    def __post_init__(self) -> None:
        Task.__post_init__(self)
        _non_empty(self.research_key, "research_key")
        if self.target_level is not None and (
            isinstance(self.target_level, bool)
            or not isinstance(self.target_level, int)
            or self.target_level < 1
        ):
            raise ValueError("target_level must be a positive integer")


@dataclass(frozen=True, slots=True)
class ProduceDrone(Task):
    """Produce one drone with the requested equipment loadout."""

    equipment: tuple[str, ...] = ()
    kind: ClassVar[str] = "produce_drone"

    def __post_init__(self) -> None:
        Task.__post_init__(self)
        equipment = tuple(self.equipment)
        if any(not isinstance(item, str) or not item.strip() for item in equipment):
            raise ValueError("equipment entries must be non-empty strings")
        object.__setattr__(self, "equipment", equipment)


TaskType: TypeAlias = (
    Task
    | MineLoop
    | Recharge
    | ScoutSector
    | Escort
    | Strike
    | LayMines
    | DeployRepeater
    | Research
    | ProduceDrone
)


__all__ = [
    "DeployRepeater",
    "Escort",
    "LayMines",
    "MineLoop",
    "ProduceDrone",
    "Recharge",
    "Research",
    "ScoutSector",
    "Strike",
    "Task",
    "TaskId",
    "TaskType",
]
