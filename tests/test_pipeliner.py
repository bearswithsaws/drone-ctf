"""Queue-saturation and invalidation tests for the planning pipeliner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agent.planning.pipeliner import Pipeliner, PlannedAction, QueueFlushError
from agent.transport.action_tracker import EntityKind, Precondition


@dataclass
class FakeResult:
    data: dict[str, Any]
    ok: bool = True
    status: int = 200


@dataclass
class FakeIntent:
    action_id: str | None


class FakeRest:
    def __init__(self) -> None:
        self.queues: dict[str, list[dict[str, Any]]] = {
            "/queue/drones": [],
            "/queue/buildings": [],
        }
        self.deletes: list[str] = []
        self.fail_deletes: set[str] = set()

    async def get(self, path: str, params=None) -> FakeResult:
        return FakeResult({"actions": [dict(action) for action in self.queues[path]]})

    async def delete(self, path: str, json=None) -> FakeResult:
        self.deletes.append(path)
        action_id = path.rsplit("/", 1)[-1]
        if action_id in self.fail_deletes:
            return FakeResult({"error": "nope"}, ok=False, status=500)
        queue_path = path.rsplit("/", 1)[0]
        self.queues[queue_path] = [
            action for action in self.queues[queue_path] if action["action_id"] != action_id
        ]
        return FakeResult({"message": "cancelled"})


class FakeTracker:
    def __init__(self, rest: FakeRest) -> None:
        self.rest = rest
        self.submissions: list[dict[str, Any]] = []
        self._sequence = 0

    async def submit(
        self,
        local_id: str,
        entity_kind: EntityKind | str,
        entity_id: str,
        action: str,
        *,
        precondition: Precondition,
        payload=None,
        distance: float = 0.0,
        endpoint: str | None = None,
    ) -> FakeIntent:
        kind = EntityKind(entity_kind)
        self._sequence += 1
        action_id = f"action-{self._sequence}"
        record = {
            "local_id": local_id,
            "entity_kind": kind,
            "entity_id": entity_id,
            "action": action,
            "payload": dict(payload or {}),
            "precondition": precondition,
            "action_id": action_id,
            "distance": distance,
            "endpoint": endpoint,
        }
        self.submissions.append(record)
        id_key = "drone_id" if kind is EntityKind.DRONE else "building_id"
        self.rest.queues[kind.queue_path].append(
            {"action_id": action_id, id_key: entity_id, "action": action}
        )
        return FakeIntent(action_id)


def actions(prefix: str, count: int = 30) -> list[PlannedAction]:
    return [
        PlannedAction("propulsion/drive", {"direction": 1, "step": f"{prefix}-{index}"})
        for index in range(count)
    ]


def make_pipeliner(**kwargs) -> tuple[Pipeliner, FakeRest, FakeTracker]:
    rest = FakeRest()
    tracker = FakeTracker(rest)
    return Pipeliner(rest, tracker, **kwargs), rest, tracker


async def test_initial_fill_reaches_target_without_exceeding_cap() -> None:
    pipeliner, rest, tracker = make_pipeliner(target_depth=6)
    pipeliner.register(EntityKind.DRONE, "drone-1", actions("old"))

    [status] = await pipeliner.cycle()

    assert status.queue_depth == 6
    assert status.submitted == 6
    assert len(rest.queues["/queue/drones"]) == 6
    assert len(tracker.submissions) == 6


async def test_queue_is_replenished_before_it_can_run_empty_across_cycles() -> None:
    pipeliner, rest, _tracker = make_pipeliner(target_depth=6)
    pipeliner.register("drone", "drone-1", actions("move"))
    await pipeliner.cycle()

    observed_depths: list[int] = []
    for _cycle in range(12):
        # The server completes one FIFO action per simulated game cycle.
        rest.queues["/queue/drones"].pop(0)
        await pipeliner.cycle()
        depth = len(rest.queues["/queue/drones"])
        observed_depths.append(depth)

    assert min(observed_depths) >= 4
    assert observed_depths == [6] * 12


async def test_fill_is_per_entity_and_preserves_another_entities_actions() -> None:
    pipeliner, rest, _tracker = make_pipeliner(target_depth=4)
    pipeliner.register("drone", "alpha", actions("a"))
    pipeliner.register("drone", "bravo", actions("b"))
    rest.queues["/queue/drones"].append(
        {"action_id": "foreign-1", "drone_id": "someone-else", "action": "status"}
    )

    statuses = await pipeliner.cycle()

    assert {(status.entity_id, status.queue_depth) for status in statuses} == {
        ("alpha", 4),
        ("bravo", 4),
    }
    assert len(rest.queues["/queue/drones"]) == 9


async def test_plan_invalidation_flushes_only_entity_and_rebuilds_immediately() -> None:
    pipeliner, rest, tracker = make_pipeliner(target_depth=4)
    pipeliner.register("drone", "alpha", actions("old"))
    pipeliner.register("drone", "bravo", actions("other"))
    await pipeliner.cycle()
    old_alpha = [item for item in tracker.submissions if item["entity_id"] == "alpha"]
    bravo_ids = {
        action["action_id"]
        for action in rest.queues["/queue/drones"]
        if action["drone_id"] == "bravo"
    }

    result = await pipeliner.replace_plan("drone", "alpha", actions("new"))

    alpha_queue = [
        action for action in rest.queues["/queue/drones"] if action["drone_id"] == "alpha"
    ]
    assert len(result.cancelled_action_ids) == 4
    assert result.rebuilt == 4
    assert result.queue_depth == 4
    assert len(alpha_queue) == 4
    assert {action["action_id"] for action in alpha_queue}.isdisjoint(
        result.cancelled_action_ids
    )
    assert {
        action["action_id"]
        for action in rest.queues["/queue/drones"]
        if action["drone_id"] == "bravo"
    } == bravo_ids
    old_preconditions = [await submission["precondition"]() for submission in old_alpha]
    assert not any(old_preconditions)
    new_alpha = [item for item in tracker.submissions if item["entity_id"] == "alpha"][-4:]
    new_preconditions = [await submission["precondition"]() for submission in new_alpha]
    assert all(new_preconditions)
    assert all(submission["payload"]["step"].startswith("new-") for submission in new_alpha)


async def test_failed_flush_stays_paused_and_does_not_mix_plans() -> None:
    pipeliner, rest, tracker = make_pipeliner(target_depth=4)
    pipeliner.register("drone", "alpha", actions("old"))
    await pipeliner.cycle()
    failed_id = rest.queues["/queue/drones"][0]["action_id"]
    rest.fail_deletes.add(failed_id)

    with pytest.raises(QueueFlushError) as exc_info:
        await pipeliner.replace_plan("drone", "alpha", actions("new"))

    assert failed_id in exc_info.value.action_ids
    assert len(tracker.submissions) == 4
    assert (await pipeliner.maintain("drone", "alpha")).paused is True
    assert len(tracker.submissions) == 4


async def test_callable_controller_can_temporarily_have_no_output() -> None:
    pipeliner, _rest, tracker = make_pipeliner(target_depth=4)
    pending: list[PlannedAction] = []

    def controller():
        return pending.pop(0) if pending else None

    pipeliner.register("drone", "alpha", controller)
    assert (await pipeliner.maintain("drone", "alpha")).queue_depth == 0

    pending.extend(actions("late", 4))
    assert (await pipeliner.maintain("drone", "alpha")).queue_depth == 4
    assert len(tracker.submissions) == 4


async def test_empty_success_response_still_counts_as_a_successful_delete() -> None:
    pipeliner, rest, _tracker = make_pipeliner(target_depth=4)
    pipeliner.register("drone", "alpha", actions("old"))
    await pipeliner.cycle()

    original_delete = rest.delete

    async def delete_with_decode_error(path: str, json=None) -> FakeResult:
        await original_delete(path, json)
        return FakeResult({"error": "empty response"}, ok=False, status=204)

    rest.delete = delete_with_decode_error  # type: ignore[method-assign]
    result = await pipeliner.replace_plan("drone", "alpha", actions("new"))

    assert len(result.cancelled_action_ids) == 4
    assert result.rebuilt == 4
    status = pipeliner.status("drone", "alpha")
    assert status is not None
    assert status.queue_depth == 4


@pytest.mark.parametrize("target", [3, 9])
def test_target_depth_is_bounded_to_superhuman_apm_window(target: int) -> None:
    rest = FakeRest()
    with pytest.raises(ValueError, match="between 4 and 8"):
        Pipeliner(rest, FakeTracker(rest), min_depth=1, target_depth=target, max_depth=10)
