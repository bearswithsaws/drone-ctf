"""Tests for the world model, tile beliefs, and change events."""

from __future__ import annotations

from agent.bus import EventBus
from agent.world import (
    TOPIC_WORLD_CHANGED,
    Source,
    Terrain,
    TileObservation,
    WorldChange,
    WorldModel,
)
from agent.world.tiles import TileBelief


# --- TileBelief (pure) ---

def test_confidence_decays_by_half_life() -> None:
    t = TileBelief(q=0, r=0, last_seen=0)
    assert t.confidence(0, half_life=100) == 1.0
    assert t.confidence(100, half_life=100) == 0.5
    assert t.confidence(200, half_life=100) == 0.25


def test_confidence_never_negative_age() -> None:
    # A tile "seen in the future" (clock skew) clamps age to 0, confidence 1.
    t = TileBelief(q=0, r=0, last_seen=500)
    assert t.age(100) == 0
    assert t.confidence(100, half_life=100) == 1.0


def test_passable_optimistic_for_unknown_terrain() -> None:
    assert TileBelief(0, 0).passable is True
    assert TileBelief(0, 0, terrain_type=Terrain.DIFFICULT).passable is True
    assert TileBelief(0, 0, terrain_type=Terrain.IMPASSABLE).passable is False


# --- WorldModel mutations + events ---

async def _events(bus: EventBus) -> list[WorldChange]:
    got: list[WorldChange] = []

    async def h(_topic: str, change: WorldChange) -> None:
        got.append(change)

    bus.subscribe(TOPIC_WORLD_CHANGED, h)
    return got


async def test_observe_tile_creates_belief_and_emits() -> None:
    bus = EventBus()
    events = await _events(bus)
    wm = WorldModel(bus)

    await wm.observe_tile(
        TileObservation(q=2, r=-1, terrain_type=1, elevation=3), cycle=10
    )
    tile = wm.get_tile((2, -1))
    assert tile is not None
    assert tile.terrain_type == 1 and tile.elevation == 3
    assert tile.last_seen == 10
    assert wm.cycle == 10
    assert events[-1].kind == "tile"
    assert events[-1].keys == ((2, -1),)


async def test_observe_tiles_batch_emits_one_event() -> None:
    bus = EventBus()
    events = await _events(bus)
    wm = WorldModel(bus)

    obs = [TileObservation(q=i, r=0, terrain_type=1) for i in range(5)]
    await wm.observe_tiles(obs, cycle=7)
    assert wm.tile_count == 5
    # Exactly one change event naming all five coords.
    tile_events = [e for e in events if e.kind == "tile"]
    assert len(tile_events) == 1
    assert len(tile_events[0].keys) == 5


async def test_partial_observation_does_not_erase_detail() -> None:
    wm = WorldModel()
    # Identify establishes terrain + resource detail.
    await wm.observe_tile(
        TileObservation(
            q=0, r=0, terrain_type=2, elevation=4,
            has_resource=True, resource_type="titanium_ore", resource_amount=500,
        ),
        cycle=1,
        source=Source.IDENTIFY,
    )
    # A later scan resolves has_resource True again but carries no type/amount.
    await wm.observe_tile(
        TileObservation(q=0, r=0, terrain_type=2, has_resource=True), cycle=5
    )
    tile = wm.get_tile((0, 0))
    assert tile.resource_type == "titanium_ore"  # preserved
    assert tile.resource_amount == 500           # preserved
    assert tile.last_seen == 5


async def test_resource_nodes_tracked_and_cleared() -> None:
    wm = WorldModel()
    await wm.observe_tile(
        TileObservation(q=1, r=1, has_resource=True, resource_type="rare_earth_ore"),
        cycle=1,
    )
    assert {t.coord for t in wm.resource_nodes()} == {(1, 1)}
    # Re-observing the tile as depleted removes it from the node set.
    await wm.observe_tile(TileObservation(q=1, r=1, has_resource=False), cycle=2)
    assert list(wm.resource_nodes()) == []


async def test_confidence_via_model_uses_current_cycle() -> None:
    wm = WorldModel(confidence_half_life=100)
    await wm.observe_tile(TileObservation(q=0, r=0), cycle=0)
    await wm.set_cycle(100)
    assert wm.confidence((0, 0)) == 0.5
    assert wm.confidence((9, 9)) == 0.0  # never seen


async def test_set_cycle_is_monotonic() -> None:
    wm = WorldModel()
    await wm.set_cycle(50)
    await wm.set_cycle(10)  # stale, ignored
    assert wm.cycle == 50


async def test_upsert_and_remove_drone() -> None:
    bus = EventBus()
    events = await _events(bus)
    wm = WorldModel(bus)

    await wm.upsert_drone("d1", q=0, r=0, direction=2, cycle=3)
    rec = wm.get_drone("d1")
    assert rec.coord == (0, 0) and rec.direction == 2
    await wm.upsert_drone("d1", q=1, r=0, cycle=4)  # move
    assert wm.get_drone("d1").coord == (1, 0)

    await wm.remove_drone("d1")
    assert wm.get_drone("d1") is None
    assert events[-1].kind == "removed"
    # removing a non-existent drone emits nothing extra
    n = len(events)
    await wm.remove_drone("ghost")
    assert len(events) == n


async def test_upsert_building_footprint() -> None:
    wm = WorldModel()
    await wm.upsert_building(
        "cc", building_type="command_center", origin=(0, 0),
        tiles=[(0, 0), (1, 0), (0, 1)], cycle=2,
    )
    rec = wm.get_building("cc")
    assert rec.building_type == "command_center"
    assert rec.origin == (0, 0)
    assert set(rec.tiles) == {(0, 0), (1, 0), (0, 1)}


async def test_enemy_buildings_are_separate_and_removable_by_observed_tile() -> None:
    wm = WorldModel()
    await wm.upsert_building("ours", building_type="laser_turret", origin=(0, 0))
    await wm.upsert_enemy_building(
        "theirs",
        building_type="missile_turret",
        origin=(10, 0),
        tiles=[(10, 0)],
        cycle=5,
    )

    assert wm.get_building("ours") is not None
    assert wm.get_enemy_building("theirs").origin == (10, 0)
    assert await wm.remove_enemy_buildings_at((10, 0)) == ("theirs",)
    assert wm.get_enemy_building("theirs") is None
    assert wm.get_building("ours") is not None


async def test_model_works_without_bus() -> None:
    wm = WorldModel()  # bus is None
    await wm.observe_tile(TileObservation(q=0, r=0, terrain_type=1), cycle=1)
    assert wm.tile_count == 1  # no crash, no emit
