"""Deterministic fitness auction for strategic tasks.

The allocator deliberately knows nothing about batteries, distance, equipment,
or doctrine.  Callers express all of those concerns in a fitness callback.  It
adds only the coordination policy shared by every controller: task capacity,
stable reassignment, and commander pins.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Generic, TypeAlias, TypeVar

from agent.planning.tasks import Task


EntityT = TypeVar("EntityT")
EntityId: TypeAlias = str
FitnessScorer: TypeAlias = Callable[[Any, Task], float | None]
EntityIdGetter: TypeAlias = Callable[[Any], str]
PinTarget: TypeAlias = str | Task


def entity_id(entity: Any) -> str:
    """Return a stable identifier for common entity representations."""

    if isinstance(entity, str):
        value = entity
    elif isinstance(entity, Mapping):
        value = next(
            (
                entity[key]
                for key in ("entity_id", "drone_id", "building_id", "id")
                if key in entity
            ),
            None,
        )
    else:
        value = next(
            (
                getattr(entity, key)
                for key in ("entity_id", "drone_id", "building_id", "id")
                if hasattr(entity, key)
            ),
            None,
        )
    if not isinstance(value, str) or not value.strip():
        raise ValueError("entity must have a non-empty entity_id, drone_id, building_id, or id")
    return value


@dataclass(frozen=True, slots=True)
class Assignment(Generic[EntityT]):
    """One winning entity/task edge from the auction."""

    entity_id: EntityId
    entity: EntityT
    task: Task
    fitness: float | None
    pinned: bool = False
    retained: bool = False

    @property
    def task_id(self) -> str:
        return self.task.task_id


@dataclass(frozen=True, slots=True)
class AllocationResult(Mapping[EntityId, Task], Generic[EntityT]):
    """Immutable auction snapshot.

    It behaves as ``entity_id -> Task`` for convenient controller lookup; the
    richer :attr:`assignments` mapping exposes scores and pin/retention flags
    for commander plan introspection.
    """

    assignments: Mapping[EntityId, Assignment[EntityT]]
    unassigned_entity_ids: tuple[EntityId, ...] = ()
    unfilled_task_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", MappingProxyType(dict(self.assignments)))

    def __getitem__(self, entity_id: EntityId) -> Task:
        return self.assignments[entity_id].task

    def __iter__(self) -> Iterator[EntityId]:
        return iter(self.assignments)

    def __len__(self) -> int:
        return len(self.assignments)

    def assignment_for(self, entity_id: EntityId) -> Assignment[EntityT] | None:
        return self.assignments.get(entity_id)

    def task_for(self, entity_id: EntityId) -> Task | None:
        assignment = self.assignment_for(entity_id)
        return assignment.task if assignment is not None else None


class TaskAllocator(Generic[EntityT]):
    """Assign tasks by greedy maximum-fitness auction.

    The auction sorts every eligible entity/task edge by descending fitness.
    Incumbent edges receive ``hysteresis`` only for winner selection, requiring
    a challenger to improve on them by that margin.  Reported fitness remains
    the unmodified score.  Input order provides deterministic tie-breaking.
    """

    def __init__(
        self,
        fitness: FitnessScorer | None = None,
        *,
        hysteresis: float = 0.1,
        min_fitness: float = 0.0,
        id_getter: EntityIdGetter = entity_id,
    ) -> None:
        if not math.isfinite(hysteresis) or hysteresis < 0:
            raise ValueError("hysteresis must be finite and non-negative")
        if not math.isfinite(min_fitness):
            raise ValueError("min_fitness must be finite")
        self._fitness = fitness
        self.hysteresis = float(hysteresis)
        self.min_fitness = float(min_fitness)
        self._id_getter = id_getter
        self._assignments: dict[EntityId, str] = {}
        self._pins: dict[EntityId, PinTarget] = {}
        self._last_result: AllocationResult[EntityT] = AllocationResult({})

    @property
    def assignments(self) -> Mapping[EntityId, str]:
        """Current ``entity_id -> task_id`` state used for hysteresis."""

        return MappingProxyType(dict(self._assignments))

    @property
    def pins(self) -> Mapping[EntityId, PinTarget]:
        return MappingProxyType(dict(self._pins))

    @property
    def last_result(self) -> AllocationResult[EntityT]:
        return self._last_result

    def pin(self, entity_id: EntityId, task: PinTarget) -> None:
        """Persistently pin an entity until :meth:`unpin` is called.

        A full ``Task`` may be pinned even when it is absent from the current
        strategist task set.  A string target is resolved on each allocation;
        while it is absent the entity stays reserved rather than violating the
        commander directive with an autonomous assignment.
        """

        if not isinstance(entity_id, str) or not entity_id.strip():
            raise ValueError("entity_id cannot be empty")
        if isinstance(task, str):
            if not task.strip():
                raise ValueError("pinned task_id cannot be empty")
        elif not isinstance(task, Task):
            raise TypeError("pinned target must be a task_id or Task")
        self._pins[entity_id] = task

    def unpin(self, entity_id: EntityId) -> None:
        self._pins.pop(entity_id, None)

    def clear_pins(self) -> None:
        self._pins.clear()

    def clear(self) -> None:
        """Forget assignment history without removing commander pins."""

        self._assignments.clear()
        self._last_result = AllocationResult({})

    def allocate(
        self,
        entities: Iterable[EntityT],
        tasks: Iterable[Task],
        fitness: FitnessScorer | None = None,
        *,
        pinned: Mapping[EntityId, PinTarget] | None = None,
    ) -> AllocationResult[EntityT]:
        """Run one auction and retain its winners for the next cycle.

        ``pinned`` is an ephemeral overlay, useful for directives whose TTL is
        managed by another component.  It takes precedence over persistent
        pins for this call and does not mutate :attr:`pins`.
        """

        scorer = fitness or self._fitness
        if scorer is None:
            raise ValueError("a fitness callback is required")

        entity_values = list(entities)
        task_values = list(tasks)
        entity_ids = [self._id_getter(value) for value in entity_values]
        self._require_unique(entity_ids, "entity")
        self._require_unique([task.task_id for task in task_values], "task")

        entities_by_id = dict(zip(entity_ids, entity_values, strict=True))
        tasks_by_id = {task.task_id: task for task in task_values}
        effective_pins = dict(self._pins)
        if pinned is not None:
            effective_pins.update(pinned)
        for pinned_entity_id, target in effective_pins.items():
            if pinned_entity_id not in entities_by_id:
                continue
            if isinstance(target, Task):
                existing = tasks_by_id.get(target.task_id)
                if existing is not None and existing != target:
                    raise ValueError(f"conflicting task definitions for {target.task_id!r}")
                tasks_by_id[target.task_id] = target

        winners: dict[EntityId, Assignment[EntityT]] = {}
        used: dict[str, int] = {task_id: 0 for task_id in tasks_by_id}

        # Commander selections are placed first and bypass both fitness and
        # capacity.  Unknown entity ids remain stored for future replans.
        for current_entity_id in entity_ids:
            if current_entity_id not in effective_pins:
                continue
            target = effective_pins[current_entity_id]
            task_id = target.task_id if isinstance(target, Task) else target
            task = tasks_by_id.get(task_id)
            if task is None:
                continue
            winners[current_entity_id] = Assignment(
                current_entity_id,
                entities_by_id[current_entity_id],
                task,
                None,
                pinned=True,
                retained=self._assignments.get(current_entity_id) == task_id,
            )
            used[task_id] += 1

        task_order = {task_id: index for index, task_id in enumerate(tasks_by_id)}
        entity_order = {current_id: index for index, current_id in enumerate(entity_ids)}
        edges: list[tuple[float, int, int, float, EntityId, Task]] = []
        for current_entity_id, value in zip(entity_ids, entity_values, strict=True):
            if current_entity_id in effective_pins:
                continue
            incumbent = self._assignments.get(current_entity_id)
            for task_id, task in tasks_by_id.items():
                score = self._score(scorer, value, task)
                if score is None or score < self.min_fitness:
                    continue
                adjusted = score + (self.hysteresis if incumbent == task_id else 0.0)
                edges.append(
                    (
                        -adjusted,
                        task_order[task_id],
                        entity_order[current_entity_id],
                        score,
                        current_entity_id,
                        task,
                    )
                )

        edges.sort(key=lambda edge: edge[:3])
        for _negative_adjusted, _task_order, _entity_order, score, current_id, task in edges:
            if current_id in winners or used[task.task_id] >= task.max_assignees:
                continue
            winners[current_id] = Assignment(
                current_id,
                entities_by_id[current_id],
                task,
                score,
                retained=self._assignments.get(current_id) == task.task_id,
            )
            used[task.task_id] += 1

        self._assignments = {
            current_entity_id: assignment.task_id
            for current_entity_id, assignment in winners.items()
        }
        unassigned = tuple(current_id for current_id in entity_ids if current_id not in winners)
        unfilled = tuple(
            task_id
            for task_id, task in tasks_by_id.items()
            if used.get(task_id, 0) < task.max_assignees
        )
        self._last_result = AllocationResult(winners, unassigned, unfilled)
        return self._last_result

    @staticmethod
    def _require_unique(values: Iterable[str], label: str) -> None:
        seen: set[str] = set()
        for value in values:
            if value in seen:
                raise ValueError(f"duplicate {label} id: {value!r}")
            seen.add(value)

    @staticmethod
    def _score(scorer: FitnessScorer, entity: EntityT, task: Task) -> float | None:
        raw = scorer(entity, task)
        if raw is None:
            return None
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError("fitness must be a number or None")
        score = float(raw) * task.priority
        if not math.isfinite(score):
            raise ValueError("fitness must be finite")
        return score


# Short names keep call sites natural and leave room for alternate allocation
# policies without changing the task framework.
Allocator = TaskAllocator
GreedyAllocator = TaskAllocator


__all__ = [
    "AllocationResult",
    "Allocator",
    "Assignment",
    "EntityId",
    "FitnessScorer",
    "GreedyAllocator",
    "PinTarget",
    "TaskAllocator",
    "entity_id",
]
