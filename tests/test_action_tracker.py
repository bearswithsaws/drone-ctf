"""Simulated-loss tests for the three-signal ActionTracker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agent.bus import EventBus
from agent.transport.action_tracker import (
    TOPIC_ACTION_LIFECYCLE,
    TOPIC_ACTION_LOSS,
    TOPIC_ACTION_SUBMITTED,
    ActionLifecycleEvent,
    ActionLossEvent,
    ActionSubmittedEvent,
    ActionTracker,
    EntityKind,
    IntentState,
    LossKind,
)


@dataclass
class FakeResult:
    data: dict[str, Any]
    ok: bool = True


class FakeRest:
    def __init__(self, action_ids: list[str | None]) -> None:
        self.action_ids = iter(action_ids)
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.gets: list[str] = []
        self.queues: dict[str, Any] = {
            "/queue/drones": [],
            "/queue/buildings": [],
        }
        self.poll_ok = True

    async def post(self, path: str, json: dict[str, Any] | None = None) -> FakeResult:
        self.posts.append((path, dict(json or {})))
        action_id = next(self.action_ids)
        if action_id is None:
            return FakeResult({"error": {"message": "connection lost"}}, ok=False)
        return FakeResult({"action_id": action_id})

    async def get(self, path: str, params: dict[str, Any] | None = None) -> FakeResult:
        self.gets.append(path)
        return FakeResult({"actions": self.queues[path]}, ok=self.poll_ok)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float = 1.0) -> None:
        self.now += seconds


@dataclass
class Harness:
    tracker: ActionTracker
    rest: FakeRest
    clock: FakeClock
    lifecycle: list[ActionLifecycleEvent]
    losses: list[ActionLossEvent]
    submissions: list[ActionSubmittedEvent]


def make_harness(action_ids: list[str | None]) -> Harness:
    rest = FakeRest(action_ids)
    clock = FakeClock()
    bus = EventBus()
    lifecycle: list[ActionLifecycleEvent] = []
    losses: list[ActionLossEvent] = []
    submissions: list[ActionSubmittedEvent] = []

    async def capture_lifecycle(_topic: str, event: ActionLifecycleEvent) -> None:
        lifecycle.append(event)

    async def capture_loss(_topic: str, event: ActionLossEvent) -> None:
        losses.append(event)

    async def capture_submission(_topic: str, event: ActionSubmittedEvent) -> None:
        submissions.append(event)

    bus.subscribe(TOPIC_ACTION_LIFECYCLE, capture_lifecycle)
    bus.subscribe(TOPIC_ACTION_LOSS, capture_loss)
    bus.subscribe(TOPIC_ACTION_SUBMITTED, capture_submission)
    tracker = ActionTracker(rest, bus, ack_timeout=lambda _distance: 1.0, clock=clock)
    return Harness(tracker, rest, clock, lifecycle, losses, submissions)


async def submit_drive(harness: Harness, *, precondition=lambda: True) -> None:
    await harness.tracker.submit(
        "move-1",
        EntityKind.DRONE,
        "drone-7",
        "propulsion/drive",
        payload={"direction": "forward", "level": 1},
        distance=12,
        precondition=precondition,
    )


async def test_drop_command_resubmits_then_reaches_completed() -> None:
    h = make_harness([None, "retry-id"])
    await submit_drive(h)

    # No queued report and the first action is absent from the authoritative queue.
    h.clock.advance()
    await h.tracker.reconcile()
    assert [path for path, _body in h.rest.posts] == [
        "/drones/drone-7/propulsion/drive",
        "/drones/drone-7/propulsion/drive",
    ]
    assert h.tracker.get("move-1").attempts == 2  # type: ignore[union-attr]
    assert [event.kind for event in h.losses] == [LossKind.COMMAND]
    assert [event.attempt for event in h.submissions] == [1, 2]
    assert h.submissions[0].payload == {"direction": "forward", "level": 1}

    await h.tracker.on_message(
        {
            "message_type": "action_queued",
            "action_id": "retry-id",
            "details": {"cycles_required": 2},
        }
    )
    await h.tracker.on_message(
        {"message_type": "action_completed", "action_id": "retry-id", "details": {}}
    )
    intent = h.tracker.get("move-1")
    assert intent is not None
    assert intent.state is IntentState.COMPLETED
    assert intent.action_id == "retry-id"


async def test_acknowledged_action_absent_from_queue_is_never_resubmitted() -> None:
    h = make_harness(["post-id"])
    await submit_drive(h)

    h.clock.advance()
    await h.tracker.reconcile()

    intent = h.tracker.get("move-1")
    assert intent is not None
    assert intent.state is IntentState.COMPLETED
    assert intent.inferred_terminal
    assert intent.acknowledged
    assert intent.post_action_id == "post-id"
    assert len(h.rest.posts) == 1
    assert [event.kind for event in h.losses] == [
        LossKind.QUEUED_REPORT,
        LossKind.COMPLETION_REPORT,
    ]
    assert h.lifecycle[-1].reason == "acknowledged_action_absent_from_queue"


async def test_successful_post_without_usable_id_is_still_acknowledged() -> None:
    h = make_harness([""])
    await submit_drive(h)

    h.clock.advance()
    await h.tracker.reconcile()

    intent = h.tracker.get("move-1")
    assert intent is not None
    assert intent.acknowledged
    assert intent.post_action_id is None
    assert intent.state is IntentState.COMPLETED
    assert len(h.rest.posts) == 1


@pytest.mark.parametrize(
    ("action", "subject", "terminal_type", "expected_state"),
    [
        ("sensors/scan", "Scan completed", "action_completed", IntentState.COMPLETED),
        ("status", "Status completed", "action_completed", IntentState.COMPLETED),
        ("propulsion/drive", "Drive completed", "action_completed", IntentState.COMPLETED),
        ("propulsion/drive", "Drive failed", "action_failed", IntentState.FAILED),
    ],
)
async def test_mismatched_post_and_lifecycle_ids_correlate_by_entity_and_action(
    action: str,
    subject: str,
    terminal_type: str,
    expected_state: IntentState,
) -> None:
    h = make_harness(["post-id"])
    await h.tracker.submit(
        "intent-1",
        EntityKind.DRONE,
        "drone-7",
        action,
        payload={"level": 1},
        precondition=lambda: True,
    )

    verb = subject.split()[0]
    await h.tracker.on_message(
        {
            "message_type": "action_queued",
            "action_id": "lifecycle-id",
            "drone_id": "drone-7",
            "subject": f"{verb} queued",
            "details": {"level": 1, "cycles_required": 1},
        }
    )
    await h.tracker.on_message(
        {
            "message_type": terminal_type,
            "action_id": "lifecycle-id",
            "drone_id": "drone-7",
            "subject": subject,
            "details": {"success": expected_state is IntentState.COMPLETED},
        }
    )

    intent = h.tracker.get("intent-1")
    assert intent is not None
    assert intent.state is expected_state
    assert intent.post_action_id == "post-id"
    assert intent.action_id == "lifecycle-id"
    assert intent.server_action_ids == ["post-id", "lifecycle-id"]
    assert intent.attempts == 1
    assert len(h.rest.posts) == 1


async def test_mismatched_queue_id_correlates_without_queued_message() -> None:
    h = make_harness(["post-id"])
    await submit_drive(h)
    h.rest.queues["/queue/drones"] = [
        {
            "action_id": "lifecycle-id",
            "drone_id": "drone-7",
            "action": "propulsion/drive",
        }
    ]

    h.clock.advance()
    await h.tracker.reconcile()

    intent = h.tracker.get("move-1")
    assert intent is not None
    assert intent.state is IntentState.QUEUED
    assert intent.action_id == "lifecycle-id"
    assert intent.server_action_ids == ["post-id", "lifecycle-id"]
    assert len(h.rest.posts) == 1


async def test_fast_terminal_message_with_mismatched_id_needs_no_queue_poll() -> None:
    h = make_harness(["post-id"])
    await h.tracker.submit(
        "scan-1",
        EntityKind.DRONE,
        "drone-7",
        "sensors/scan",
        payload={"level": 1},
        precondition=lambda: True,
    )

    # The action completed before the first poll and its queued message was lost.
    await h.tracker.on_message(
        {
            "message_type": "action_completed",
            "action_id": "lifecycle-id",
            "drone_id": "drone-7",
            "subject": "Scan completed",
            "details": {"success": True},
        }
    )
    h.clock.advance()
    await h.tracker.reconcile()

    intent = h.tracker.get("scan-1")
    assert intent is not None
    assert intent.state is IntentState.COMPLETED
    assert not intent.inferred_terminal
    assert intent.action_id == "lifecycle-id"
    assert len(h.rest.posts) == 1
    assert h.rest.gets == []


async def test_repeated_actions_use_payload_then_fifo_for_mismatched_ids() -> None:
    h = make_harness(["post-1", "post-2"])
    for local_id, direction in (("move-1", 1), ("move-2", -1)):
        await h.tracker.submit(
            local_id,
            EntityKind.DRONE,
            "drone-7",
            "propulsion/drive",
            payload={"direction": direction, "level": 1},
            precondition=lambda: True,
        )

    # The second action's queued report arrives first. Its echoed payload must
    # prevent the lifecycle ID from being attached to the first FIFO candidate.
    await h.tracker.on_message(
        {
            "message_type": "action_queued",
            "action_id": "lifecycle-2",
            "drone_id": "drone-7",
            "subject": "Drive queued",
            "details": {"direction": -1, "level": 1, "cycles_required": 1},
        }
    )
    await h.tracker.on_message(
        {
            "message_type": "action_queued",
            "action_id": "lifecycle-1",
            "drone_id": "drone-7",
            "subject": "Drive queued",
            "details": {"direction": 1, "level": 1, "cycles_required": 1},
        }
    )
    for lifecycle_id in ("lifecycle-1", "lifecycle-2"):
        await h.tracker.on_message(
            {
                "message_type": "action_completed",
                "action_id": lifecycle_id,
                "drone_id": "drone-7",
                "subject": "Drive completed",
                "details": {"success": True},
            }
        )

    first = h.tracker.get("move-1")
    second = h.tracker.get("move-2")
    assert first is not None and second is not None
    assert first.action_id == "lifecycle-1"
    assert second.action_id == "lifecycle-2"
    assert first.state is second.state is IntentState.COMPLETED


async def test_unknown_lifecycle_id_does_not_cross_entity_or_action() -> None:
    h = make_harness(["post-id"])
    await submit_drive(h)

    for drone_id, subject in (("other-drone", "Drive completed"), ("drone-7", "Scan completed")):
        await h.tracker.on_message(
            {
                "message_type": "action_completed",
                "action_id": f"foreign-{drone_id}-{subject}",
                "drone_id": drone_id,
                "subject": subject,
                "details": {"success": True},
            }
        )

    intent = h.tracker.get("move-1")
    assert intent is not None
    assert intent.state is IntentState.SUBMITTED
    assert intent.server_action_ids == ["post-id"]


async def test_orphaned_queued_alias_is_replayed_when_terminal_subject_correlates() -> None:
    h = make_harness(["post-id"])
    await h.tracker.submit(
        "repeater-1",
        EntityKind.DRONE,
        "drone-7",
        "repeater/activate",
        precondition=lambda: True,
    )

    # This live subject is not derivable from the endpoint action, but queued
    # and terminal reports share the lifecycle ID. The terminal subject creates
    # the alias and must replay the earlier queued report.
    await h.tracker.on_message(
        {
            "message_type": "action_queued",
            "action_id": "lifecycle-id",
            "drone_id": "drone-7",
            "subject": "Repeater transmit queued",
            "details": {"cycles_required": 1},
        }
    )
    await h.tracker.on_message(
        {
            "message_type": "action_completed",
            "action_id": "lifecycle-id",
            "drone_id": "drone-7",
            "subject": "Repeater Activate completed",
            "details": {"success": True},
        }
    )

    intent = h.tracker.get("repeater-1")
    assert intent is not None
    assert intent.state is IntentState.COMPLETED
    assert intent.saw_queued
    assert not h.losses


async def test_drop_queued_report_is_recovered_from_queue_without_resubmit() -> None:
    h = make_harness(["action-1"])
    await submit_drive(h)
    h.rest.queues["/queue/drones"] = [{"action_id": "action-1", "drone_id": "drone-7"}]

    h.clock.advance()
    await h.tracker.reconcile()
    intent = h.tracker.get("move-1")
    assert intent is not None
    assert intent.state is IntentState.QUEUED
    assert len(h.rest.posts) == 1
    assert [event.kind for event in h.losses] == [LossKind.QUEUED_REPORT]

    await h.tracker.on_message(
        {"message_type": "action_completed", "action_id": "action-1", "details": {}}
    )
    assert h.tracker.get("move-1").state is IntentState.COMPLETED  # type: ignore[union-attr]


async def test_drop_completed_report_is_inferred_when_action_leaves_queue() -> None:
    h = make_harness(["action-1"])
    await submit_drive(h)
    await h.tracker.on_message(
        {
            "message_type": "action_queued",
            "action_id": "action-1",
            "details": {"cycles_required": 2},
        }
    )

    # 2 cycles x .25s plus the 1s ack window.
    h.clock.advance(1.5)
    await h.tracker.reconcile()
    intent = h.tracker.get("move-1")
    assert intent is not None
    assert intent.state is IntentState.COMPLETED
    assert intent.inferred_terminal is True
    assert [event.kind for event in h.losses] == [LossKind.COMPLETION_REPORT]
    assert h.lifecycle[-1].reason == "queue_poll_action_disappeared"


async def test_changed_precondition_abandons_without_double_submission() -> None:
    h = make_harness([None])
    still_holds = False
    await submit_drive(h, precondition=lambda: still_holds)

    h.clock.advance()
    await h.tracker.reconcile()
    intent = h.tracker.get("move-1")
    assert intent is not None
    assert intent.state is IntentState.ABANDONED
    assert intent.terminal
    assert len(h.rest.posts) == 1
    assert h.lifecycle[-1].reason == "precondition_changed"


async def test_completed_without_queued_infers_queued_report_loss() -> None:
    h = make_harness(["action-1"])
    await submit_drive(h)
    await h.tracker.on_message(
        {"message_type": "action_completed", "action_id": "action-1", "details": {}}
    )

    assert h.tracker.get("move-1").state is IntentState.COMPLETED  # type: ignore[union-attr]
    assert [event.kind for event in h.losses] == [LossKind.QUEUED_REPORT]
    assert h.lifecycle[-1].previous_state is IntentState.SUBMITTED


async def test_failed_action_reaches_failed_terminal_state() -> None:
    h = make_harness(["action-1"])
    await submit_drive(h)
    await h.tracker.on_message(
        {"message_type": "action_queued", "action_id": "action-1", "details": {}}
    )
    await h.tracker.on_message(
        {
            "message_type": "action_failed",
            "action_id": "action-1",
            "details": {"error": "occupied"},
        }
    )

    intent = h.tracker.get("move-1")
    assert intent is not None
    assert intent.state is IntentState.FAILED
    assert intent.last_message is not None
    assert intent.last_message["details"]["error"] == "occupied"


async def test_building_actions_poll_the_building_queue() -> None:
    h = make_harness(["build-1"])
    await h.tracker.submit(
        "base-status",
        EntityKind.BUILDING,
        "base-3",
        "common/status",
        precondition=lambda: True,
    )
    h.rest.queues["/queue/buildings"] = {
        "base-3": [{"action_id": "build-1"}]
    }

    h.clock.advance()
    await h.tracker.reconcile()
    assert h.rest.gets == ["/queue/buildings"]
    assert h.tracker.get("base-status").state is IntentState.QUEUED  # type: ignore[union-attr]


async def test_duplicate_local_id_is_idempotent_but_conflicts_are_rejected() -> None:
    h = make_harness(["action-1"])
    await submit_drive(h)
    await submit_drive(h)
    assert len(h.rest.posts) == 1

    with pytest.raises(ValueError, match="already belongs"):
        await h.tracker.submit(
            "move-1",
            EntityKind.DRONE,
            "drone-7",
            "propulsion/turn",
            precondition=lambda: True,
        )


async def test_message_arriving_before_post_response_is_replayed() -> None:
    h = make_harness(["action-1"])
    await h.tracker.on_message(
        {"message_type": "action_completed", "action_id": "action-1", "details": {}}
    )
    await submit_drive(h)

    intent = h.tracker.get("move-1")
    assert intent is not None
    assert intent.state is IntentState.COMPLETED
    assert [event.kind for event in h.losses] == [LossKind.QUEUED_REPORT]


async def test_queue_poll_failure_does_not_infer_loss_or_resubmit() -> None:
    h = make_harness(["action-1"])
    await submit_drive(h)
    h.rest.poll_ok = False

    h.clock.advance()
    await h.tracker.reconcile()
    assert len(h.rest.posts) == 1
    assert not h.losses
    assert h.tracker.get("move-1").state is IntentState.SUBMITTED  # type: ignore[union-attr]
    assert h.lifecycle[-1].reason == "queue_poll_failed"


async def test_late_failure_corrects_an_inferred_completion() -> None:
    h = make_harness(["action-1"])
    await submit_drive(h)
    await h.tracker.on_message(
        {"message_type": "action_queued", "action_id": "action-1", "details": {}}
    )
    h.clock.advance()
    await h.tracker.reconcile()
    assert h.tracker.get("move-1").state is IntentState.COMPLETED  # type: ignore[union-attr]

    await h.tracker.on_message(
        {"message_type": "action_failed", "action_id": "action-1", "details": {}}
    )
    intent = h.tracker.get("move-1")
    assert intent is not None
    assert intent.state is IntentState.FAILED
    assert intent.inferred_terminal is False


async def test_async_precondition_is_supported() -> None:
    h = make_harness([None, "action-2"])

    async def holds() -> bool:
        return True

    await submit_drive(h, precondition=holds)
    h.clock.advance()
    await h.tracker.reconcile()
    assert len(h.rest.posts) == 2


async def test_distance_is_exposed_on_loss_for_comms_calibration() -> None:
    h = make_harness([None, "action-2"])
    await submit_drive(h)
    h.clock.advance()
    await h.tracker.reconcile()
    assert h.losses[0].distance == 12


def test_entity_kind_and_state_values_are_stable_for_metrics() -> None:
    assert EntityKind.DRONE.value == "drone"
    assert IntentState.COMPLETED.value == "completed"
    assert LossKind.COMMAND.value == "command_lost"
