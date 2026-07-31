"""Fitness auction, hysteresis, capacity, and commander override tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent.planning.allocator import TaskAllocator, entity_id
from agent.planning.tasks import Task


@dataclass(frozen=True)
class Candidate:
    entity_id: str
    skills: dict[str, float]


def score(candidate: Candidate, task: Task) -> float | None:
    return candidate.skills.get(task.task_id)


def test_greedy_auction_assigns_each_entity_to_best_available_task() -> None:
    alpha = Candidate("alpha", {"mine": 10, "scout": 5})
    bravo = Candidate("bravo", {"mine": 8, "scout": 9})
    mine = Task("mine")
    scout = Task("scout")

    result = TaskAllocator(score).allocate([alpha, bravo], [mine, scout])

    assert result["alpha"] is mine
    assert result["bravo"] is scout
    assert result.assignments["alpha"].fitness == 10
    assert result.unassigned_entity_ids == ()
    assert result.unfilled_task_ids == ()


def test_capacity_and_priority_participate_in_the_auction() -> None:
    alpha = Candidate("alpha", {"shared": 6, "priority": 4})
    bravo = Candidate("bravo", {"shared": 5, "priority": 1})
    shared = Task("shared", max_assignees=2)
    priority = Task("priority", priority=2)

    result = TaskAllocator(score).allocate([alpha, bravo], [shared, priority])

    assert result["alpha"] is priority
    assert result["bravo"] is shared
    assert result.assignments["alpha"].fitness == 8
    assert result.unfilled_task_ids == ("shared",)


def test_none_and_below_threshold_scores_leave_entities_unassigned() -> None:
    alpha = Candidate("alpha", {"bad": -1})
    tasks = [Task("missing"), Task("bad")]

    result = TaskAllocator(score).allocate([alpha], tasks)

    assert dict(result) == {}
    assert result.unassigned_entity_ids == ("alpha",)
    assert result.unfilled_task_ids == ("missing", "bad")


def test_hysteresis_suppresses_small_score_oscillations_then_allows_clear_win() -> None:
    skills = {"mine": 10.0, "scout": 9.0}
    alpha = Candidate("alpha", skills)
    mine = Task("mine")
    scout = Task("scout")
    allocator = TaskAllocator(score, hysteresis=0.25)

    assert allocator.allocate([alpha], [mine, scout])["alpha"] is mine

    skills.update(mine=9.9, scout=10.0)
    stable = allocator.allocate([alpha], [mine, scout])
    assert stable["alpha"] is mine
    assert stable.assignments["alpha"].retained is True

    skills.update(mine=9.9, scout=10.2)
    switched = allocator.allocate([alpha], [mine, scout])
    assert switched["alpha"] is scout
    assert switched.assignments["alpha"].retained is False


def test_ties_are_deterministic_and_do_not_oscillate_with_reordered_entities() -> None:
    alpha = Candidate("alpha", {"task": 1})
    bravo = Candidate("bravo", {"task": 1})
    task = Task("task")
    allocator = TaskAllocator(score, hysteresis=0.1)

    assert allocator.allocate([alpha, bravo], [task])["alpha"] is task
    second = allocator.allocate([bravo, alpha], [task])
    assert second["alpha"] is task
    assert "bravo" not in second


def test_persistent_pin_overrides_fitness_and_reserves_capacity() -> None:
    alpha = Candidate("alpha", {"mine": 100, "scout": 0})
    bravo = Candidate("bravo", {"mine": 90, "scout": 5})
    mine = Task("mine")
    scout = Task("scout")
    allocator = TaskAllocator(score)
    allocator.pin("alpha", "scout")

    result = allocator.allocate([alpha, bravo], [mine, scout])

    assert result["alpha"] is scout
    assert result.assignments["alpha"].pinned is True
    assert result.assignments["alpha"].fitness is None
    assert result["bravo"] is mine

    allocator.unpin("alpha")
    assert allocator.allocate([alpha, bravo], [mine, scout])["alpha"] is mine


def test_multiple_commander_pins_override_task_capacity() -> None:
    alpha = Candidate("alpha", {})
    bravo = Candidate("bravo", {})
    task = Task("hold", max_assignees=1)
    allocator = TaskAllocator(score)

    result = allocator.allocate(
        [alpha, bravo], [task], pinned={"alpha": "hold", "bravo": "hold"}
    )

    assert set(result) == {"alpha", "bravo"}
    assert all(assignment.pinned for assignment in result.assignments.values())


def test_ephemeral_pin_can_supply_override_task_without_persisting_it() -> None:
    alpha = Candidate("alpha", {"autonomy": 5})
    autonomy = Task("autonomy")
    override = Task("commander-hold")
    allocator = TaskAllocator(score)

    pinned = allocator.allocate([alpha], [autonomy], pinned={"alpha": override})
    assert pinned["alpha"] is override
    assert allocator.pins == {}
    assert allocator.allocate([alpha], [autonomy])["alpha"] is autonomy


def test_unresolved_pin_reserves_entity_until_directive_task_returns() -> None:
    alpha = Candidate("alpha", {"autonomy": 5, "override": 0})
    allocator = TaskAllocator(score)
    allocator.pin("alpha", "override")

    waiting = allocator.allocate([alpha], [Task("autonomy")])
    assert "alpha" not in waiting
    assert waiting.unassigned_entity_ids == ("alpha",)

    override = Task("override")
    assert allocator.allocate([alpha], [Task("autonomy"), override])["alpha"] is override


def test_default_entity_identifier_handles_world_records_and_mappings() -> None:
    @dataclass
    class Drone:
        drone_id: str

    assert entity_id(Drone("drone-1")) == "drone-1"
    assert entity_id({"building_id": "plant-1"}) == "plant-1"
    assert entity_id("raw-id") == "raw-id"
    with pytest.raises(ValueError):
        entity_id(object())


def test_duplicate_ids_and_invalid_fitness_are_rejected() -> None:
    alpha = Candidate("alpha", {"task": 1})
    allocator = TaskAllocator(score)

    with pytest.raises(ValueError, match="duplicate entity"):
        allocator.allocate([alpha, alpha], [Task("task")])
    with pytest.raises(ValueError, match="duplicate task"):
        allocator.allocate([alpha], [Task("task"), Task("task")])
    with pytest.raises(ValueError, match="finite"):
        TaskAllocator(lambda _entity, _task: float("nan")).allocate([alpha], [Task("task")])
