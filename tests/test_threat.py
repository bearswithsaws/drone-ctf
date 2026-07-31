"""Threat-map envelope, confidence, and update tests."""

from __future__ import annotations

import pytest

from agent.world import (
    Sighting,
    SightingSource,
    ThreatMap,
    TrackStore,
    WorldModel,
)


async def _threat_map(*, half_life: int = 120) -> tuple[WorldModel, TrackStore, ThreatMap]:
    world = WorldModel()
    tracks = TrackStore(half_life=half_life, gate=10)
    return world, tracks, ThreatMap(world, tracks)


@pytest.mark.parametrize(
    ("building_type", "inside", "outside"),
    [
        ("laser_turret", (15, 0), (16, 0)),
        ("missile_turret", (15, 0), (16, 0)),
        ("auto_cannon_turret", (22, 0), (23, 0)),
    ],
)
async def test_turret_ranges_are_inclusive(
    building_type: str, inside: tuple[int, int], outside: tuple[int, int]
) -> None:
    world, _, threat = await _threat_map()
    await world.upsert_enemy_building(
        "turret", building_type=building_type, origin=(0, 0), cycle=1
    )

    assert threat.cost_at(inside) == 1.0
    assert threat.cost_at(outside) == 0.0


async def test_howitzer_has_eight_to_thirty_two_tile_envelope() -> None:
    _, tracks, threat = await _threat_map()
    await tracks.observe(
        Sighting(
            SightingSource.IDENTIFY,
            0,
            0,
            1,
            drone_id="artillery",
            equipment=frozenset({"howitzer"}),
        )
    )

    assert threat.cost_at((7, 0), now_cycle=1) == 0.0
    assert threat.cost_at((8, 0), now_cycle=1) == 1.0
    assert threat.cost_at((32, 0), now_cycle=1) == 1.0
    assert threat.cost_at((33, 0), now_cycle=1) == 0.0


async def test_identified_laser_drone_threatens_to_ten_tiles() -> None:
    _, tracks, threat = await _threat_map()
    await tracks.observe(
        Sighting(
            SightingSource.IDENTIFY,
            0,
            0,
            1,
            equipment=frozenset({"laser_cannon"}),
        )
    )

    assert threat.cost_at((10, 0), now_cycle=1) == 1.0
    assert threat.cost_at((11, 0), now_cycle=1) == 0.0


async def test_scan_only_contact_is_conservatively_treated_as_laser_threat() -> None:
    _, tracks, threat = await _threat_map()
    await tracks.observe(Sighting(SightingSource.SCAN, 0, 0, 1))

    source = threat.sources(now_cycle=1)[0]
    assert source.weapon == "laser_cannon"
    assert threat.cost_at((10, 0), now_cycle=1) == 1.0


async def test_identify_can_prove_contact_noncombat_or_decoy() -> None:
    _, tracks, threat = await _threat_map()
    await tracks.observe(
        Sighting(SightingSource.IDENTIFY, 0, 0, 1, equipment=frozenset())
    )
    await tracks.observe(
        Sighting(
            SightingSource.IDENTIFY,
            20,
            0,
            1,
            is_decoy=True,
            equipment=frozenset({"laser_cannon"}),
        )
    )

    assert threat.sources(now_cycle=1) == ()


async def test_track_cost_decays_with_confidence() -> None:
    _, tracks, threat = await _threat_map(half_life=100)
    await tracks.observe(Sighting(SightingSource.SCAN, 0, 0, 0))

    assert threat.cost_at((5, 0), now_cycle=0) == 1.0
    assert threat.cost_at((5, 0), now_cycle=100) == pytest.approx(0.5)


async def test_moving_track_envelope_uses_predicted_position() -> None:
    _, tracks, threat = await _threat_map()
    await tracks.observe(Sighting(SightingSource.SCAN, 0, 0, 0))
    await tracks.observe(Sighting(SightingSource.SCAN, 2, 0, 2))

    source = threat.sources(now_cycle=4)[0]
    assert source.coord == (4, 0)
    assert source.cost_at((14, 0)) > 0
    assert source.cost_at((15, 0)) == 0


async def test_overlapping_envelopes_add_and_batch_matches_single_queries() -> None:
    world, tracks, threat = await _threat_map()
    await tracks.observe(Sighting(SightingSource.SCAN, 0, 0, 1))
    await world.upsert_enemy_building(
        "laser-turret", building_type="laser_turret", origin=(5, 0), cycle=1
    )

    coords = [(3, 0), (30, 0)]
    assert threat.cost_at((3, 0), now_cycle=1) == 2.0
    assert threat.costs_at(coords, now_cycle=1) == {(3, 0): 2.0, (30, 0): 0.0}


async def test_building_add_move_and_remove_are_visible_without_rebuilding_map() -> None:
    world, _, threat = await _threat_map()
    assert threat((0, 0)) == 0.0

    await world.upsert_enemy_building(
        "turret", building_type="laser_turret", origin=(0, 0), cycle=1
    )
    assert threat((0, 0)) == 1.0

    await world.upsert_enemy_building("turret", origin=(30, 0), cycle=2)
    assert threat((0, 0)) == 0.0
    assert threat((30, 0)) == 1.0

    await world.remove_enemy_building("turret")
    assert threat((30, 0)) == 0.0
