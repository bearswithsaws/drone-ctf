"""Ingest-registry tests: wire messages -> world model.

Uses the version-controlled golden corpus for realistic status_report/scan
payloads, plus synthetic messages for movement, destruction, and the
scan-before-position buffering fix.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.world import Ingestor, WorldModel

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "message_samples.json").read_text(encoding="utf-8")
)


def sample(subject: str) -> dict:
    for m in GOLDEN:
        if m.get("subject") == subject:
            return m
    raise AssertionError(f"no golden sample with subject {subject!r}")


def _drive(drone_id: str, direction: str = "forward", cycle: int = 1) -> dict:
    return {
        "message_id": f"drive-{cycle}",
        "message_type": "action_completed",
        "subject": "Drive completed",
        "timestamp": cycle,
        "drone_id": drone_id,
        "details": {"success": True, "drive_direction": direction},
    }


def _turn(drone_id: str, new_direction: int, cycle: int = 1) -> dict:
    return {
        "message_id": f"turn-{cycle}",
        "message_type": "action_completed",
        "subject": "Turn completed",
        "timestamp": cycle,
        "drone_id": drone_id,
        "details": {"success": True, "new_direction": new_direction},
    }


# --- status_report ---

async def test_status_report_populates_buildings_and_drones() -> None:
    wm = WorldModel()
    ing = Ingestor(wm)
    await ing.ingest(sample("Building status_report completed"))

    buildings = list(wm.buildings())
    drones = list(wm.drones())
    assert len(buildings) == 8
    assert len(drones) == 2

    cc = wm.get_building(
        next(b.building_id for b in buildings if b.building_type == "command_center")
    )
    assert cc.origin == (0, 0)
    # CC footprint tiles are marked as buildings in the tile layer.
    assert wm.get_tile((0, 0)).has_building is True
    # A reported drone is placed at its absolute location.
    assert any(d.coord is not None for d in drones)


async def test_status_report_removes_stale_entities() -> None:
    wm = WorldModel()
    ing = Ingestor(wm)
    # Seed a drone/building that won't be in the report.
    await wm.upsert_drone("ghost-drone", q=9, r=9, cycle=0)
    await wm.upsert_building("ghost-building", origin=(9, 9), cycle=0)

    await ing.ingest(sample("Building status_report completed"))
    assert wm.get_drone("ghost-drone") is None
    assert wm.get_building("ghost-building") is None


# --- scan + buffering fix ---

async def test_scan_buffered_until_position_known_then_replayed() -> None:
    wm = WorldModel()
    ing = Ingestor(wm)
    scan = sample("Scan completed")
    scan_drone = scan["drone_id"]

    # Position unknown -> scan is buffered, not dropped, no tiles yet.
    await ing.ingest(scan)
    assert wm.tile_count == 0

    # A status_report that places the scanning drone flushes the buffer.
    report = json.loads(json.dumps(sample("Building status_report completed")))
    report["details"]["drones"].append({"drone_id": scan_drone, "location": [10, 10]})
    await ing.ingest(report)

    assert wm.tile_count > 0
    # The scan's own tile (relative (0,0), distance 0 if present) or any tile is
    # now anchored at the drone origin (10, 10) + relative offset.
    # tile[0] in the sample is relative (-1,-1) -> absolute (9, 9).
    assert wm.get_tile((9, 9)) is not None


async def test_scan_applied_directly_when_position_known() -> None:
    wm = WorldModel()
    ing = Ingestor(wm)
    scan = sample("Scan completed")
    await wm.upsert_drone(scan["drone_id"], q=0, r=0, cycle=0)
    await ing.ingest(scan)
    # 45 tiles in the sample scan, anchored at origin (0,0).
    assert wm.tile_count == 45
    assert wm.get_tile((-1, -1)) is not None  # relative (-1,-1) + origin (0,0)


# --- movement ---

async def test_drive_moves_drone_in_facing_direction() -> None:
    wm = WorldModel()
    ing = Ingestor(wm)
    # Facing 2 (S) at (0,0); forward drive -> neighbor in direction 2.
    await wm.upsert_drone("d1", q=0, r=0, direction=2, cycle=0)
    from agent.rules import hexmath

    expected = hexmath.get_neighbor(0, 0, 2)
    await ing.ingest(_drive("d1", "forward", cycle=5))
    assert wm.get_drone("d1").coord == expected


async def test_reverse_drive_uses_opposite_direction() -> None:
    wm = WorldModel()
    ing = Ingestor(wm)
    await wm.upsert_drone("d1", q=0, r=0, direction=0, cycle=0)
    from agent.rules import hexmath

    expected = hexmath.get_neighbor(0, 0, 3)  # opposite of 0
    await ing.ingest(_drive("d1", "reverse", cycle=5))
    assert wm.get_drone("d1").coord == expected


async def test_drive_ignored_without_known_direction() -> None:
    wm = WorldModel()
    ing = Ingestor(wm)
    await wm.upsert_drone("d1", q=0, r=0, cycle=0)  # no direction
    await ing.ingest(_drive("d1", "forward", cycle=5))
    assert wm.get_drone("d1").coord == (0, 0)  # unchanged


async def test_turn_updates_direction() -> None:
    wm = WorldModel()
    ing = Ingestor(wm)
    await wm.upsert_drone("d1", q=0, r=0, direction=0, cycle=0)
    await ing.ingest(_turn("d1", 4, cycle=3))
    assert wm.get_drone("d1").direction == 4


# --- destruction + fallthrough ---

async def test_drone_destroyed_removes_it() -> None:
    wm = WorldModel()
    ing = Ingestor(wm)
    await wm.upsert_drone("d1", q=0, r=0, cycle=0)
    await ing.ingest(
        {"message_id": "x", "message_type": "drone_destroyed", "subject": "Drone destroyed",
         "timestamp": 9, "drone_id": "d1", "details": {}}
    )
    assert wm.get_drone("d1") is None


async def test_unmapped_message_is_noop() -> None:
    wm = WorldModel()
    ing = Ingestor(wm)
    await ing.ingest(
        {"message_id": "x", "message_type": "score_update", "subject": "Score changed",
         "timestamp": 1, "details": {"new_score": 5}}
    )
    assert wm.tile_count == 0 and not list(wm.drones())


async def test_on_message_bus_entrypoint_ignores_non_dict() -> None:
    wm = WorldModel()
    ing = Ingestor(wm)
    await ing.on_message("message", "not a dict")  # must not raise
    assert wm.tile_count == 0
