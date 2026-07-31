"""Recorded-session regression tests for the replay harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.replay import (
    ReplayFormatError,
    ReplayMismatch,
    assert_expected_snapshot,
    load_expected_snapshot,
    replay_recording,
)

FIXTURES = Path(__file__).parent / "fixtures"
SESSION = FIXTURES / "replay_session.jsonl"
EXPECTED = FIXTURES / "replay_expected.json"


async def test_recorded_session_reaches_known_world_and_entity_state() -> None:
    result = await replay_recording(SESSION)

    assert result.records_seen == 7
    assert result.messages_replayed == 6
    assert result.snapshot == load_expected_snapshot(EXPECTED)

    world = result.snapshot["world"]
    assert world["cycle"] == 15
    assert world["drones"] == [
        {
            "drone_id": "scout-1",
            "q": 2,
            "r": -1,
            "direction": 2,
            "elevation": 1,
            "last_seen": 15,
        }
    ]
    assert world["enemy_tracks"][0]["drone_id"] == "enemy-1"
    assert world["enemy_buildings"][0]["building_id"] == "enemy-turret"

    drone = result.snapshot["entity_sim"]["drones"][0]
    # Scan (1), identify (1), and turn (2) are debited after the 100-unit checkpoint.
    assert drone["current_battery"] == 96
    assert drone["consumables"] == {"missile_launcher": 2}


async def test_malformed_jsonl_reports_the_line(tmp_path: Path) -> None:
    recording = tmp_path / "broken.jsonl"
    recording.write_text('{"version":1}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ReplayFormatError, match=r"broken\.jsonl:2: invalid JSON"):
        await replay_recording(recording)


def test_snapshot_mismatch_contains_a_unified_diff() -> None:
    expected = {"world": {"cycle": 10}}
    actual = {"world": {"cycle": 11}}

    with pytest.raises(ReplayMismatch, match=r"-\s+\"cycle\": 10") as raised:
        assert_expected_snapshot(actual, expected, expected_path="expected.json")

    assert "replayed state" in str(raised.value)


def test_expected_snapshot_must_be_an_object(tmp_path: Path) -> None:
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ReplayFormatError, match="must be a JSON object"):
        load_expected_snapshot(expected)
