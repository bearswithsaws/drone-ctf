"""Telemetry metric counter and gauge tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent.bus import EventBus
from agent.telemetry.metrics import (
    TOPIC_BATTERY_MARGIN,
    TOPIC_KILL_DEATH,
    TOPIC_QUEUE_DEPTH,
    BatteryMarginEvent,
    KillDeathEvent,
    Metrics,
    QueueDepthEvent,
)
from agent.transport.action_tracker import (
    TOPIC_ACTION_LOSS,
    TOPIC_ACTION_SUBMITTED,
    ActionLossEvent,
    ActionSubmittedEvent,
    ActionTracker,
    EntityKind,
    LossKind,
)


async def test_loss_rates_increment_in_the_correct_distance_bucket() -> None:
    bus = EventBus()
    metrics = Metrics(bus, distance_buckets=(10, 20))

    for index, distance in enumerate((4.0, 10.0, 19.9, 25.0), start=1):
        await bus.publish(
            TOPIC_ACTION_SUBMITTED,
            ActionSubmittedEvent(
                local_id=f"action-{index}",
                entity_kind=EntityKind.DRONE,
                entity_id="drone-1",
                action="scan",
                endpoint="/scan",
                payload={},
                distance=distance,
                attempt=1,
                timestamp=1.0,
            ),
        )

    await bus.publish(
        TOPIC_ACTION_LOSS,
        ActionLossEvent(
            local_id="action-2",
            entity_kind=EntityKind.DRONE,
            entity_id="drone-1",
            action="scan",
            action_id="server-2",
            kind=LossKind.QUEUED_REPORT,
            distance=10.0,
            attempt=1,
            timestamp=2.0,
        ),
    )
    await bus.publish(
        TOPIC_ACTION_LOSS,
        {
            "kind": "completion_report_lost",
            "distance": 19.9,
        },
    )

    buckets = metrics.loss_rate_by_distance
    assert (buckets["0-10"].submissions, buckets["0-10"].losses) == (1, 0)
    assert (buckets["10-20"].submissions, buckets["10-20"].losses) == (2, 2)
    assert buckets["10-20"].loss_rate == 1.0
    assert buckets["10-20"].losses_by_kind == {
        "queued_report_lost": 1,
        "completion_report_lost": 1,
    }
    assert (buckets["20+"].submissions, buckets["20+"].losses) == (1, 0)


@dataclass
class FakeResult:
    data: dict[str, Any]
    ok: bool = True


class LossyRest:
    def __init__(self) -> None:
        self.posts = 0

    async def post(self, path: str, json: dict[str, Any] | None = None) -> FakeResult:
        self.posts += 1
        return FakeResult({"action_id": f"server-{self.posts}"})

    async def get(self, path: str, params: dict[str, Any] | None = None) -> FakeResult:
        return FakeResult({"actions": []})


class Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_action_tracker_simulated_loss_feeds_metrics() -> None:
    bus = EventBus()
    metrics = Metrics(bus, distance_buckets=(10, 20))
    clock = Clock()
    tracker = ActionTracker(
        LossyRest(),
        bus,
        clock=clock,
        ack_timeout=lambda _distance: 1.0,
    )

    await tracker.submit(
        "move-1",
        EntityKind.DRONE,
        "drone-1",
        "propulsion/drive",
        distance=12,
        precondition=lambda: True,
    )
    clock.now = 1.0
    await tracker.reconcile()  # absent from queue: command lost, then resubmitted

    bucket = metrics.loss_rate_by_distance["10-20"]
    assert bucket.submissions == 2
    assert bucket.losses == 1
    assert bucket.loss_rate == 0.5
    assert bucket.losses_by_kind == {"command_lost": 1}


async def test_queue_combat_and_battery_events_update_snapshots() -> None:
    bus = EventBus()
    metrics = Metrics(bus)

    await bus.publish(TOPIC_QUEUE_DEPTH, QueueDepthEvent("drone", "drone-1", 4))
    await bus.publish(
        TOPIC_QUEUE_DEPTH, {"entity_kind": "building", "entity_id": "base", "depth": 2}
    )
    await bus.publish(
        TOPIC_KILL_DEATH,
        KillDeathEvent("drone-1", "enemy-1", timestamp=12.0, action_id="shot-1"),
    )
    await bus.publish(
        TOPIC_KILL_DEATH, {"killer_id": "drone-1", "victim_id": "enemy-2", "timestamp": 13.0}
    )
    await bus.publish(
        TOPIC_BATTERY_MARGIN,
        BatteryMarginEvent(
            "drone-1",
            current_battery=45,
            required_battery=20,
            reserve_battery=5,
            max_battery=100,
        ),
    )

    assert metrics.queue_depths == {"drone": {"drone-1": 4}, "building": {"base": 2}}
    assert metrics.kill_death_ledger["drone-1"].kills == 2
    assert metrics.kill_death_ledger["enemy-1"].deaths == 1
    assert metrics.battery_margins["drone-1"].margin == 20
    assert metrics.battery_margins["drone-1"].margin_fraction == 0.2
    json.dumps(metrics.snapshot())


def test_invalid_bucket_edges_and_queue_depth_are_rejected() -> None:
    try:
        Metrics(distance_buckets=(10, 10))
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("duplicate bucket edge was accepted")

    metrics = Metrics()
    try:
        metrics.observe_queue_depth("drone", "drone-1", -1)
    except ValueError as exc:
        assert "cannot be negative" in str(exc)
    else:
        raise AssertionError("negative queue depth was accepted")
