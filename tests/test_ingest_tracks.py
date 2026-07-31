"""Ingest → TrackStore integration: scans/identifies produce enemy tracks."""

from __future__ import annotations

from agent.world import Ingestor, TrackStore, WorldModel


def _scan_msg(drone_id: str, tiles: list[dict], cycle: int = 1) -> dict:
    return {
        "message_id": f"scan-{cycle}",
        "message_type": "action_completed",
        "subject": "Scan completed",
        "timestamp": cycle,
        "drone_id": drone_id,
        "details": {"scan_data": {"tiles": tiles}},
    }


async def test_scan_seeds_enemy_track_for_foreign_drone() -> None:
    wm = WorldModel()
    ts = TrackStore()
    ing = Ingestor(wm, ts)
    # Our scanning drone is at (0,0).
    await wm.upsert_drone("me", q=0, r=0, cycle=0)
    # Scan sees a drone at relative (3,0) -> absolute (3,0), not ours.
    await ing.ingest(
        _scan_msg("me", [{"coordinates": [3, 0], "terrain_type": 1, "elevation": 0,
                          "has_resource": False, "has_drone": True, "has_building": False}])
    )
    assert ts.count == 1
    assert ts.tracks()[0].coord == (3, 0)


async def test_scan_ignores_our_own_drone() -> None:
    wm = WorldModel()
    ts = TrackStore()
    ing = Ingestor(wm, ts)
    await wm.upsert_drone("me", q=0, r=0, cycle=0)
    await wm.upsert_drone("friend", q=2, r=0, cycle=0)  # our drone at (2,0)
    # Scan sees a drone at absolute (2,0) — that's our friend, not an enemy.
    await ing.ingest(
        _scan_msg("me", [{"coordinates": [2, 0], "terrain_type": 1, "elevation": 0,
                          "has_resource": False, "has_drone": True, "has_building": False}])
    )
    assert ts.count == 0


async def test_identify_attaches_identity_and_decoy() -> None:
    wm = WorldModel()
    ts = TrackStore(gate=3)
    ing = Ingestor(wm, ts)
    await wm.upsert_drone("me", q=0, r=0, cycle=0)
    # First a scan seeds a position-only track at (4,0).
    await ing.ingest(
        _scan_msg("me", [{"coordinates": [4, 0], "terrain_type": 1, "elevation": 0,
                          "has_resource": False, "has_drone": True, "has_building": False}], cycle=1)
    )
    # Then identify that tile: real enemy drone with an id.
    await ing.ingest({
        "message_id": "id-1", "message_type": "action_completed", "subject": "Identify completed",
        "timestamp": 2, "drone_id": "me",
        "details": {"success": True, "identify_data": {
            "target_q": 4, "target_r": 0, "terrain": 1, "elevation": 0, "resource": None,
            "drones": [{"drone_id": "enemy-1", "max_health": 100, "equipment": [1, 2, 3]}],
        }},
    })
    assert ts.count == 1
    t = ts.tracks()[0]
    assert t.drone_id == "enemy-1"
    assert t.is_decoy is False


async def test_ingest_without_trackstore_is_fine() -> None:
    wm = WorldModel()
    ing = Ingestor(wm)  # no track store
    await wm.upsert_drone("me", q=0, r=0, cycle=0)
    await ing.ingest(
        _scan_msg("me", [{"coordinates": [3, 0], "terrain_type": 1, "elevation": 0,
                          "has_resource": False, "has_drone": True, "has_building": False}])
    )
    # No crash; tile still recorded.
    assert wm.get_tile((3, 0)) is not None
