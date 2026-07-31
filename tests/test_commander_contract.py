from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from agent.commander.contract import (
    CONTRACT_VERSION,
    ContractV1,
    Directive,
    EntityOverrideDirective,
    StateSnapshot,
)


def test_state_snapshot_is_json_native_and_versioned() -> None:
    snapshot = StateSnapshot(
        contract_version=CONTRACT_VERSION,
        sequence=4,
        cycle=25,
        generated_at=datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc),
        tiles=[],
        drones=[],
        buildings=[],
        enemy_tracks=[],
    )

    payload = json.loads(snapshot.model_dump_json())

    assert payload["contract_version"] == "1.0"
    assert payload["generated_at"] == "2026-07-31T12:00:00Z"
    assert ContractV1.model_validate(payload).root == snapshot


def test_directive_union_uses_kind_discriminator() -> None:
    directive = TypeAdapter(Directive).validate_python(
        {
            "kind": "entity_override",
            "directive_id": "directive-1",
            "issued_at": "2026-07-31T12:00:00Z",
            "ttl_seconds": 30,
            "entity_id": "drone-1",
            "task_id": "scout-north",
        }
    )

    assert directive == EntityOverrideDirective(
        kind="entity_override",
        directive_id="directive-1",
        issued_at=datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc),
        ttl_seconds=30,
        entity_id="drone-1",
        task_id="scout-north",
    )


def test_contract_rejects_unknown_fields_and_invalid_version() -> None:
    payload = {
        "contract_version": "2.0",
        "sequence": 0,
        "cycle": 0,
        "generated_at": "2026-07-31T12:00:00Z",
        "tiles": [],
        "drones": [],
        "buildings": [],
        "enemy_tracks": [],
        "surprise": True,
    }

    with pytest.raises(ValidationError):
        StateSnapshot.model_validate(payload)


def test_json_schema_contains_all_public_contract_categories() -> None:
    definitions = ContractV1.model_json_schema()["$defs"]

    assert {
        "StateSnapshot",
        "WorldDiff",
        "PlanSnapshot",
        "StanceDirective",
        "SquadOrderDirective",
        "EntityOverrideDirective",
        "WorldDiffEvent",
        "PlanChangedEvent",
        "AlertEvent",
    } <= definitions.keys()
