"""EnemyTrack / TrackStore fusion tests."""

from __future__ import annotations

from agent.world.tracks import (
    BearingSighting,
    Sighting,
    SightingSource,
    TrackStore,
)


def _scan(q: int, r: int, cycle: int) -> Sighting:
    return Sighting(SightingSource.SCAN, q, r, cycle)


async def test_new_sighting_creates_track() -> None:
    ts = TrackStore()
    t = await ts.observe(_scan(5, 5, 10))
    assert ts.count == 1
    assert t.coord == (5, 5)
    assert t.first_seen == 10 and t.last_seen == 10


async def test_nearby_sighting_updates_same_track() -> None:
    ts = TrackStore(gate=3)
    await ts.observe(_scan(5, 5, 10))
    await ts.observe(_scan(6, 5, 12))  # 1 tile away -> same track
    assert ts.count == 1
    t = ts.tracks()[0]
    assert t.coord == (6, 5)
    assert t.last_seen == 12


async def test_far_sighting_creates_second_track() -> None:
    ts = TrackStore(gate=3)
    await ts.observe(_scan(5, 5, 10))
    await ts.observe(_scan(20, 20, 11))  # outside gate
    assert ts.count == 2


async def test_identity_association_beats_proximity() -> None:
    ts = TrackStore(gate=3)
    # Two identified drones far apart.
    await ts.observe(Sighting(SightingSource.IDENTIFY, 0, 0, 1, drone_id="A"))
    await ts.observe(Sighting(SightingSource.IDENTIFY, 30, 30, 1, drone_id="B"))
    # A later sighting of A far from its last spot still binds to A by id.
    await ts.observe(Sighting(SightingSource.IDENTIFY, 25, 30, 5, drone_id="A"))
    assert ts.count == 2
    a = next(t for t in ts.tracks() if t.drone_id == "A")
    assert a.coord == (25, 30)


async def test_scan_then_identify_attaches_identity() -> None:
    ts = TrackStore(gate=3)
    await ts.observe(_scan(5, 5, 10))
    await ts.observe(Sighting(SightingSource.IDENTIFY, 5, 5, 11, drone_id="A", is_decoy=True))
    assert ts.count == 1
    t = ts.tracks()[0]
    assert t.drone_id == "A"
    assert t.is_decoy is True


async def test_velocity_and_prediction() -> None:
    ts = TrackStore(gate=5)
    await ts.observe(_scan(0, 0, 0))
    await ts.observe(_scan(2, 0, 2))  # +1 q per cycle
    t = ts.tracks()[0]
    assert t.velocity() == (1.0, 0.0)
    assert t.predict(4) == (4, 0)  # last (2,0) + v*(4-2)


async def test_confidence_decays_and_prunes() -> None:
    ts = TrackStore(half_life=100, min_confidence=0.4, max_age=10_000)
    await ts.observe(_scan(0, 0, 0))
    t = ts.tracks()[0]
    assert t.confidence(0) == 1.0
    assert t.confidence(100) == 0.5
    removed = await ts.prune(now_cycle=200)  # confidence 0.25 < 0.4
    assert removed == [t.track_id]
    assert ts.count == 0


async def test_prune_hard_age_cap() -> None:
    ts = TrackStore(half_life=10_000, min_confidence=0.0, max_age=50)
    await ts.observe(_scan(0, 0, 0))
    assert await ts.prune(now_cycle=40) == []
    assert len(await ts.prune(now_cycle=60)) == 1


async def test_out_of_order_sighting_does_not_rewind_position() -> None:
    ts = TrackStore(gate=5)
    await ts.observe(_scan(5, 5, 20))
    await ts.observe(_scan(1, 1, 5))  # older, within gate
    t = ts.tracks()[0]
    assert t.coord == (5, 5)  # kept the newer position
    assert t.last_seen == 20


async def test_bearings_stored_separately() -> None:
    ts = TrackStore()
    await ts.observe_bearing(BearingSighting(SightingSource.SEISMIC, 0, 0, 2, cycle=5))
    assert ts.count == 0  # bearings don't create point tracks
    assert len(ts.recent_bearings()) == 1


async def test_nearest_track() -> None:
    ts = TrackStore(gate=1)
    await ts.observe(_scan(0, 0, 1))
    await ts.observe(_scan(10, 0, 1))
    near = ts.nearest((9, 0))
    assert near.coord == (10, 0)
