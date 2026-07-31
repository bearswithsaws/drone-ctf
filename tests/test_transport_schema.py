"""Golden tests for typed game-server wire messages."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from agent.transport.parsing import parse_wire_message
from agent.transport.schema import (
    ActionCompletedMessage,
    ActionFailedMessage,
    ActionQueuedMessage,
    ActionRejectedMessage,
    BuildingActionCompletedMessage,
    BuildingActionQueuedMessage,
    BuildingStatusCompletedDetails,
    BuildingStatusReportCompletedDetails,
    DroneStatusCompletedDetails,
    InfoMessage,
    ScanCompletedDetails,
    UnknownWireMessage,
    WarningMessage,
    WireModel,
)

GOLDEN_SAMPLES = Path(__file__).parent / "fixtures" / "message_samples.json"
LIVE_BUILDING_VARIANTS = (
    Path(__file__).parent / "fixtures" / "live_building_status_variants.json"
)

MESSAGE_MODELS = {
    "action_completed": ActionCompletedMessage,
    "action_failed": ActionFailedMessage,
    "action_queued": ActionQueuedMessage,
    "action_rejected": ActionRejectedMessage,
    "building_action_completed": BuildingActionCompletedMessage,
    "building_action_queued": BuildingActionQueuedMessage,
    "info": InfoMessage,
    "warning": WarningMessage,
}


def test_every_golden_sample_parses_to_a_known_typed_model() -> None:
    samples = json.loads(GOLDEN_SAMPLES.read_text(encoding="utf-8"))
    assert len(samples) == 29

    for payload in samples:
        message = parse_wire_message(payload)
        description = f'{payload["message_type"]}: {payload["subject"]}'
        assert not isinstance(message, UnknownWireMessage), description
        assert isinstance(message, MESSAGE_MODELS[payload["message_type"]]), description
        assert isinstance(message.details, WireModel), description


def test_large_nested_payloads_are_typed() -> None:
    samples = json.loads(GOLDEN_SAMPLES.read_text(encoding="utf-8"))

    def sample(subject: str) -> dict[str, object]:
        return next(payload for payload in samples if payload["subject"] == subject)

    scan = parse_wire_message(sample("Scan completed"))
    assert isinstance(scan, ActionCompletedMessage)
    assert isinstance(scan.details, ScanCompletedDetails)
    assert scan.details.scan_data.tiles[0].coordinates == (-1, -1)
    assert scan.details.scan_data.summary.total_tiles == 45

    status = parse_wire_message(sample("Status completed"))
    assert isinstance(status, ActionCompletedMessage)
    assert isinstance(status.details, DroneStatusCompletedDetails)
    assert status.details.status.equipment[1].equipment_type == "drill"
    assert status.details.status.equipment[1].efficiency == 0.15

    building_status = parse_wire_message(sample("Building status completed"))
    assert isinstance(building_status, BuildingActionCompletedMessage)
    assert isinstance(building_status.details, BuildingStatusCompletedDetails)
    assert building_status.details.status.coordinates[0] == (0, 0)

    report = parse_wire_message(sample("Building status_report completed"))
    assert isinstance(report, BuildingActionCompletedMessage)
    assert isinstance(report.details, BuildingStatusReportCompletedDetails)
    assert report.details.buildings[0].origin == (0, 0)


def test_live_building_status_variants_are_typed_without_warnings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    samples = json.loads(LIVE_BUILDING_VARIANTS.read_text(encoding="utf-8"))

    with caplog.at_level(logging.WARNING, logger="agent.transport.parsing"):
        messages = [parse_wire_message(payload) for payload in samples]

    assert not caplog.text
    assert all(isinstance(message, BuildingActionCompletedMessage) for message in messages)
    production = messages[0].details
    refiner = messages[1].details
    assert isinstance(production, BuildingStatusCompletedDetails)
    assert isinstance(refiner, BuildingStatusCompletedDetails)
    assert production.status.construction_queue == []
    assert production.status.resource_capacity is None
    assert production.status.drone_pending is False
    assert refiner.status.resource_capacity == 200
    assert refiner.status.unload_range == 1
    assert refiner.status.transfer_rate == 10


def test_unknown_message_is_logged_and_preserved(
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = {
        "message_id": "future-1",
        "message_type": "quantum_entanglement_detected",
        "timestamp": 42,
        "subject": "Unexpected signal",
        "details": {"spectrum": [1, 2, 3], "nested": {"kept": True}},
        "server_extension": "also kept",
    }

    with caplog.at_level(logging.WARNING, logger="agent.transport.parsing"):
        message = parse_wire_message(payload)

    assert isinstance(message, UnknownWireMessage)
    assert message.raw_payload == payload
    assert message.model_dump(exclude_unset=True) == payload
    assert "quantum_entanglement_detected" in caplog.text
    assert repr(payload) in caplog.text


def test_malformed_known_message_is_not_dropped(caplog: pytest.LogCaptureFixture) -> None:
    payload = {"message_type": "action_completed", "details": {"future": "shape"}}

    with caplog.at_level(logging.WARNING, logger="agent.transport.parsing"):
        message = parse_wire_message(payload)

    assert isinstance(message, UnknownWireMessage)
    assert message.raw_payload == payload
    assert "Unrecognised action_completed wire payload" in caplog.text
