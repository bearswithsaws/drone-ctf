"""Queue ETA forecasts against observed and synthetic replay sequences."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.sim.eta import (
    EntityState,
    PendingAction,
    completion_predictions,
    normalise_action,
    predict_pipeline,
    predict_queue,
)
from agent.transport.parsing import parse_wire_message

FIXTURES = Path(__file__).parent / "fixtures" / "message_samples.json"


def queued(
    action_id: str,
    subject: str,
    cycles: int,
    **details: object,
) -> dict[str, object]:
    return {
        "message_type": "action_queued",
        "action_id": action_id,
        "drone_id": "drone-1",
        "subject": subject,
        "details": {**details, "cycles_required": cycles},
    }


def test_fifo_forecast_changes_state_only_on_each_completion_cycle() -> None:
    initial = EntityState(position=(0, 0), heading=0, battery=100, total_weight=41)
    pipeline = [
        queued("turn", "Turn queued", 2, direction="right", level=3),
        queued("drive", "Drive queued", 3, direction="forward", level=2),
        queued("reverse", "Drive queued", 1, direction=-1, level=1),
    ]

    forecast = predict_pipeline(
        initial,
        pipeline,
        start_cycle=40,
        research_tiers={"propulsion": 1},
    )

    assert [item.cycle for item in forecast] == [41, 42, 43, 44, 45, 46]
    assert [item.action_id for item in forecast] == [
        "turn",
        "turn",
        "drive",
        "drive",
        "drive",
        "reverse",
    ]

    # Turn cost: int(10 * .85) = 8.  It completes facing SE.
    assert (forecast[0].position, forecast[0].heading, forecast[0].battery) == (
        (0, 0),
        0,
        100,
    )
    assert (forecast[1].position, forecast[1].heading, forecast[1].battery) == (
        (0, 0),
        1,
        92,
    )

    # Drive cost: int(10 * .85) + ceil(41/20) = 11.  Odd-q movement uses
    # heading 1 from the even starting column, so the destination is (1, 0).
    assert all(item.position == (0, 0) and item.battery == 92 for item in forecast[2:4])
    assert (forecast[4].position, forecast[4].heading, forecast[4].battery) == (
        (1, 0),
        1,
        81,
    )

    # Reverse moves opposite heading 1.  On odd q, direction 4 returns to the
    # origin.  Its level-1 cost is int(5 * .85) + 3 = 7.
    assert (forecast[5].position, forecast[5].heading, forecast[5].battery) == (
        (0, 0),
        1,
        74,
    )
    assert [item.completes_action for item in forecast] == [False, True, False, False, True, True]


def test_non_movement_actions_hold_pose_and_debit_battery_at_completion() -> None:
    forecast = predict_queue(
        EntityState((4, 3), heading=5, battery=3),
        [
            PendingAction("Scan queued", 2, {"level": 1}, "scan"),
            PendingAction("Status queued", 1, {}, "status"),
            PendingAction("Future action queued", 1, {}, "unknown"),
        ],
    )

    assert [(item.position, item.heading) for item in forecast] == [((4, 3), 5)] * 4
    assert [item.battery for item in forecast] == [3, 2, 1, 1]
    assert [item.action_id for item in completion_predictions(forecast)] == [
        "scan",
        "status",
        "unknown",
    ]


def test_battery_never_falls_below_zero() -> None:
    forecast = predict_pipeline(
        EntityState((0, 0), heading=0, battery=2, total_weight=100),
        [queued("drive", "Drive queued", 1, direction="forward", level=3)],
    )

    assert forecast[-1].battery == 0
    assert forecast[-1].position == (1, -1)


def test_raw_rest_queue_variants_and_partially_elapsed_head_are_supported() -> None:
    action = normalise_action(
        {
            "id": "rest-action",
            "action_type": "propulsion/turn",
            "cycles_required": 9,
            "cycles_remaining": 2,
            "parameters": {"direction": -1, "level": 1},
        }
    )

    assert action == PendingAction(
        subject="propulsion/turn",
        cycles_required=2,
        details={"direction": -1, "level": 1},
        action_id="rest-action",
    )
    forecast = predict_pipeline(EntityState((3, 3), 0, 10), [action])
    assert len(forecast) == 2
    assert forecast[-1].heading == 5


def test_typed_wire_message_can_be_forecast_directly() -> None:
    typed = parse_wire_message(
        {
            "message_id": "message-1",
            "message_type": "action_queued",
            "timestamp": 10,
            "subject": "Drive queued",
            "details": {"direction": "forward", "level": 1, "cycles_required": 1},
            "action_id": "action-1",
            "drone_id": "drone-1",
        }
    )

    forecast = predict_pipeline(EntityState((1, 0), heading=0, battery=10), [typed])

    assert forecast[-1].position == (2, 0)  # heading 0 from an odd column
    assert forecast[-1].battery == 5


def test_observed_queue_durations_predict_replay_completion_cycles() -> None:
    records = json.loads(FIXTURES.read_text(encoding="utf-8"))
    by_action: dict[str, dict[str, dict[str, object]]] = {}
    for record in records:
        action_id = record.get("action_id")
        message_type = record.get("message_type")
        if isinstance(action_id, str) and message_type in {"action_queued", "action_completed"}:
            by_action.setdefault(action_id, {})[message_type] = record

    # The fixture contains successful drive and turn queue/completion pairs.
    # Server timestamps are inclusive: an N-cycle action queued at T completes
    # at T + N - 1.  Seeding the forecast at T - 1 reproduces that completion.
    movement_pairs = [
        pair
        for pair in by_action.values()
        if "action_queued" in pair
        and "action_completed" in pair
        and pair["action_queued"]["subject"] in {"Drive queued", "Turn queued"}
    ]
    assert {pair["action_queued"]["subject"] for pair in movement_pairs} == {
        "Drive queued",
        "Turn queued",
    }

    for pair in movement_pairs:
        queued_message = pair["action_queued"]
        completed_message = pair["action_completed"]
        start_cycle = int(queued_message["timestamp"]) - 1
        final = predict_pipeline(
            EntityState((0, 0), heading=0, battery=4_000),
            [queued_message],
            start_cycle=start_cycle,
        )[-1]

        assert final.cycle == completed_message["timestamp"]
        if queued_message["subject"] == "Turn queued":
            assert final.heading == completed_message["details"]["new_direction"]


@pytest.mark.parametrize(
    ("action", "error"),
    [
        ({"subject": "Scan queued", "details": {}}, "cycles"),
        ({"subject": "Scan queued", "details": {"cycles_required": 0}}, "positive"),
        ({"details": {"cycles_required": 1}}, "subject"),
    ],
)
def test_invalid_queue_items_fail_loudly(action: dict[str, object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        normalise_action(action)


@pytest.mark.parametrize("heading", [-1, 6])
def test_invalid_initial_heading_is_rejected(heading: int) -> None:
    with pytest.raises(ValueError, match="heading"):
        EntityState((0, 0), heading, 10)
