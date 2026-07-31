"""Append-only session recorder tests."""

from __future__ import annotations

import json
from pathlib import Path

from agent.bus import EventBus
from agent.telemetry.recorder import RECORDING_VERSION, JsonlRecorder
from agent.transport.action_tracker import (
    TOPIC_ACTION_SUBMITTED,
    ActionSubmittedEvent,
    EntityKind,
)
from agent.transport.ws import TOPIC_MESSAGE


async def test_records_inbound_messages_and_actions_in_replayable_envelopes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sessions" / "match.jsonl"
    bus = EventBus()
    recorder = JsonlRecorder(
        path,
        bus,
        clock=lambda: 1234.5,
        session_id="session-7",
    )

    message = {
        "message_id": "msg-1",
        "message_type": "action_completed",
        "action_id": "action-1",
        "details": {"success": True},
    }
    await bus.publish(TOPIC_MESSAGE, message)
    await bus.publish(
        TOPIC_ACTION_SUBMITTED,
        ActionSubmittedEvent(
            local_id="move-1",
            entity_kind=EntityKind.DRONE,
            entity_id="drone-1",
            action="propulsion/drive",
            endpoint="/drones/drone-1/propulsion/drive",
            payload={"direction": "forward", "level": 1},
            distance=12.0,
            attempt=1,
            timestamp=99.0,
        ),
    )
    await recorder.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [
        {
            "version": RECORDING_VERSION,
            "session_id": "session-7",
            "sequence": 1,
            "recorded_at": 1234.5,
            "topic": "message",
            "payload": message,
        },
        {
            "version": RECORDING_VERSION,
            "session_id": "session-7",
            "sequence": 2,
            "recorded_at": 1234.5,
            "topic": "action.submitted",
            "payload": {
                "local_id": "move-1",
                "entity_kind": "drone",
                "entity_id": "drone-1",
                "action": "propulsion/drive",
                "endpoint": "/drones/drone-1/propulsion/drive",
                "payload": {"direction": "forward", "level": 1},
                "distance": 12.0,
                "attempt": 1,
                "timestamp": 99.0,
            },
        },
    ]


async def test_reopening_a_recording_appends_without_truncating(tmp_path: Path) -> None:
    path = tmp_path / "match.jsonl"
    path.write_text('{"from":"older-session"}\n')

    recorder = JsonlRecorder(path, clock=lambda: 1.0, session_id="new-session")
    await recorder.record(TOPIC_MESSAGE, {"message_type": "info", "details": {}})
    await recorder.close()

    lines = path.read_text().splitlines()
    assert json.loads(lines[0]) == {"from": "older-session"}
    assert json.loads(lines[1])["session_id"] == "new-session"


async def test_close_unsubscribes_recorder(tmp_path: Path) -> None:
    path = tmp_path / "match.jsonl"
    bus = EventBus()
    recorder = JsonlRecorder(path, bus)
    await bus.publish(TOPIC_MESSAGE, {"message_id": "before"})
    await recorder.close()
    await bus.publish(TOPIC_MESSAGE, {"message_id": "after"})

    assert len(path.read_text().splitlines()) == 1
