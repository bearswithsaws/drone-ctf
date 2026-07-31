"""Network-free tests for the E1.7 hardcoded autonomy loop."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from agent.bus import EventBus
from agent.planning.controllers.hello_world import HelloWorldController
from agent.transport.action_tracker import (
    TOPIC_ACTION_LIFECYCLE,
    ActionLifecycleEvent,
    ActionTracker,
    IntentState,
)
from agent.transport.ws import TOPIC_MESSAGE


@dataclass
class FakeIntent:
    local_id: str
    state: IntentState
    action_id: str = "server-action"
    attempts: int = 1
    inferred_terminal: bool = False

    @property
    def terminal(self) -> bool:
        return self.state.terminal


class FakeTracker:
    def __init__(self, outcomes: list[IntentState]) -> None:
        self.outcomes = iter(outcomes)
        self.submissions: list[dict[str, Any]] = []
        self.intents: dict[str, FakeIntent] = {}

    async def submit(
        self,
        local_id: str,
        entity_kind: object,
        entity_id: str,
        action: str,
        *,
        payload: dict[str, int],
        distance: float,
        precondition: Callable[[], bool],
    ) -> FakeIntent:
        assert precondition()
        intent = FakeIntent(local_id, next(self.outcomes))
        self.intents[local_id] = intent
        self.submissions.append(
            {
                "local_id": local_id,
                "entity_kind": entity_kind,
                "entity_id": entity_id,
                "action": action,
                "payload": payload,
                "distance": distance,
            }
        )
        return intent

    def get(self, local_id: str) -> FakeIntent | None:
        return self.intents.get(local_id)


@dataclass
class FakeResult:
    data: dict[str, Any]
    ok: bool = True


class CompletingRest:
    """Fake server that reports queued/completed through the real event bus."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.tasks: list[asyncio.Task[None]] = []

    async def post(self, path: str, json: dict[str, Any] | None = None) -> FakeResult:
        self.posts.append((path, dict(json or {})))
        action_id = f"action-{len(self.posts)}"

        async def report_lifecycle() -> None:
            # Let ActionTracker finish binding the POST response's action_id
            # before delivering the asynchronous server notifications.
            await asyncio.sleep(0)
            await self.bus.publish(
                TOPIC_MESSAGE,
                {
                    "message_type": "action_queued",
                    "action_id": action_id,
                    "details": {"cycles_required": 1},
                },
            )
            await self.bus.publish(
                TOPIC_MESSAGE,
                {
                    "message_type": "action_completed",
                    "action_id": action_id,
                    "details": {"success": True},
                },
            )

        self.tasks.append(asyncio.create_task(report_lifecycle()))
        return FakeResult({"action_id": action_id})

    async def get(self, path: str, params: dict[str, Any] | None = None) -> FakeResult:
        assert path == "/queue/drones"
        return FakeResult({"actions": []})


async def test_scan_three_tiles_and_return_are_all_terminal_and_sequential() -> None:
    tracker = FakeTracker([IntentState.COMPLETED] * 7)
    controller = HelloWorldController(tracker, "drone-7")  # type: ignore[arg-type]

    result = await controller.run_once()

    assert result.scan_state is IntentState.COMPLETED
    assert result.outbound_moves == 3
    assert result.return_moves == 3
    assert result.returned
    assert controller.displacement == 0
    assert [submission["action"] for submission in tracker.submissions] == [
        "sensors/scan",
        "propulsion/drive",
        "propulsion/drive",
        "propulsion/drive",
        "propulsion/drive",
        "propulsion/drive",
        "propulsion/drive",
    ]
    assert [submission["payload"].get("direction") for submission in tracker.submissions] == [
        None,
        1,
        1,
        1,
        -1,
        -1,
        -1,
    ]
    assert [submission["distance"] for submission in tracker.submissions] == [
        0.0,
        0.0,
        1.0,
        2.0,
        3.0,
        2.0,
        1.0,
    ]
    assert len({submission["local_id"] for submission in tracker.submissions}) == 7


async def test_real_action_tracker_observes_queued_and_completed_for_every_action() -> None:
    bus = EventBus()
    rest = CompletingRest(bus)
    tracker = ActionTracker(rest, bus, reconcile_interval_s=0.001)  # type: ignore[arg-type]
    lifecycle: dict[str, list[IntentState]] = {}

    async def capture(_topic: str, event: ActionLifecycleEvent) -> None:
        lifecycle.setdefault(event.local_id, []).append(event.state)

    bus.subscribe(TOPIC_ACTION_LIFECYCLE, capture)
    controller = HelloWorldController(tracker, "drone-7", poll_interval_s=0.001)

    result = await controller.run_once()
    await asyncio.gather(*rest.tasks)

    assert result.returned
    assert len(rest.posts) == 7
    assert len(lifecycle) == 7
    assert all(IntentState.QUEUED in states for states in lifecycle.values())
    assert all(states[-1] is IntentState.COMPLETED for states in lifecycle.values())
    assert all(intent.terminal for intent in tracker.intents.values())
    tracker.close()


async def test_failed_outbound_action_returns_only_successful_distance() -> None:
    tracker = FakeTracker(
        [
            IntentState.COMPLETED,  # scan
            IntentState.COMPLETED,  # first outbound tile
            IntentState.FAILED,  # second outbound tile
            IntentState.COMPLETED,  # reverse the one successful tile
        ]
    )
    controller = HelloWorldController(tracker, "drone-7")  # type: ignore[arg-type]

    result = await controller.run_once()

    assert result.outbound_moves == 1
    assert result.return_moves == 1
    assert result.returned
    assert controller.displacement == 0
    assert [submission["payload"].get("direction") for submission in tracker.submissions] == [
        None,
        1,
        1,
        -1,
    ]


async def test_failed_return_is_retried_before_another_scan() -> None:
    tracker = FakeTracker(
        [
            IntentState.COMPLETED,  # scan
            IntentState.COMPLETED,
            IntentState.COMPLETED,
            IntentState.COMPLETED,
            IntentState.FAILED,  # first return attempt
            IntentState.FAILED,  # next loop's recovery attempt
        ]
    )
    controller = HelloWorldController(tracker, "drone-7")  # type: ignore[arg-type]

    first = await controller.run_once()
    second = await controller.run_once()

    assert not first.returned
    assert second.scan_state is IntentState.ABANDONED
    assert controller.displacement == 3
    assert [submission["payload"].get("direction") for submission in tracker.submissions[-2:]] == [
        -1,
        -1,
    ]


async def test_non_terminal_action_times_out_instead_of_issuing_the_next_action() -> None:
    tracker = FakeTracker([IntentState.SUBMITTED])
    controller = HelloWorldController(
        tracker,  # type: ignore[arg-type]
        "drone-7",
        action_timeout_s=0.01,
        poll_interval_s=0.001,
    )

    with pytest.raises(TimeoutError, match="did not reach a terminal state"):
        await controller.run_once()

    assert len(tracker.submissions) == 1


async def test_zero_duration_does_not_start_a_partial_loop() -> None:
    tracker = FakeTracker([])
    controller = HelloWorldController(tracker, "drone-7")  # type: ignore[arg-type]

    assert await controller.run(duration_s=0) == 0
    assert tracker.submissions == []
