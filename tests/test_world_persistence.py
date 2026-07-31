"""SQLite checkpoint/restore tests for the in-memory world model."""

from __future__ import annotations

import asyncio
import sqlite3
import time

import pytest

from agent.bus import EventBus
from agent.world import (
    Source,
    TileObservation,
    WorldChange,
    WorldModel,
    WorldPersistence,
)
from agent.world import persistence as persistence_module


async def _populated_world(bus: EventBus | None = None) -> WorldModel:
    world = WorldModel(bus, confidence_half_life=73)
    await world.observe_tile(
        TileObservation(
            q=-2,
            r=7,
            terrain_type=2,
            elevation=4,
            has_resource=True,
            resource_type="rare_earth_ore",
            resource_amount=321,
            has_building=True,
            building_id="enemy-hq",
            has_drone=True,
        ),
        cycle=41,
        source=Source.IDENTIFY,
    )
    await world.upsert_drone(
        "ours-1", q=3, r=-1, direction=5, elevation=10, cycle=42
    )
    await world.upsert_building(
        "base-1",
        building_type="command_center",
        origin=(0, 0),
        tiles=((0, 0), (1, 0), (0, 1)),
        cycle=40,
    )
    await world.upsert_enemy_building(
        "enemy-hq",
        building_type="command_center",
        origin=(-2, 7),
        tiles=((-2, 7), (-1, 7)),
        cycle=41,
    )
    return world


async def test_round_trip_restores_identical_beliefs(tmp_path) -> None:
    database = tmp_path / "world.sqlite3"
    original = await _populated_world()
    expected = original.snapshot_state()

    writer = WorldPersistence(database, match_id="match-a", world=original)
    metadata = await writer.snapshot()

    restored = WorldModel(confidence_half_life=999)
    reader = WorldPersistence(database, match_id="match-a", world=restored)
    assert await reader.restore() is True
    assert restored.snapshot_state() == expected
    assert {tile.coord for tile in restored.resource_nodes()} == {(-2, 7)}
    assert metadata.cycle == 42
    assert reader.last_snapshot == metadata


async def test_snapshots_are_isolated_by_match_id(tmp_path) -> None:
    database = tmp_path / "world.sqlite3"
    first = WorldModel()
    await first.observe_tile(TileObservation(1, 1, terrain_type=1), cycle=5)
    second = WorldModel()
    await second.observe_tile(TileObservation(9, 9, terrain_type=3), cycle=8)

    await WorldPersistence(database, match_id="one", world=first).snapshot()
    await WorldPersistence(database, match_id="two", world=second).snapshot()

    restored = WorldModel()
    assert await WorldPersistence(database, match_id="one", world=restored).restore()
    assert restored.get_tile((1, 1)) is not None
    assert restored.get_tile((9, 9)) is None
    assert restored.cycle == 5


async def test_latest_snapshot_replaces_removed_beliefs(tmp_path) -> None:
    database = tmp_path / "world.sqlite3"
    world = await _populated_world()
    store = WorldPersistence(database, match_id="match", world=world)
    await store.snapshot()
    await world.remove_building("base-1")
    await store.snapshot()

    restored = WorldModel()
    assert await WorldPersistence(database, match_id="match", world=restored).restore()
    assert restored.get_building("base-1") is None


async def test_restore_emits_one_aggregate_change(tmp_path) -> None:
    database = tmp_path / "world.sqlite3"
    world = await _populated_world()
    await WorldPersistence(database, match_id="match", world=world).snapshot()

    bus = EventBus()
    changes: list[WorldChange] = []

    async def record(_topic: str, change: WorldChange) -> None:
        changes.append(change)

    bus.subscribe("world.changed", record)
    restored = WorldModel(bus)
    assert await WorldPersistence(database, match_id="match", world=restored).restore()
    assert changes == [WorldChange("restored", (), 42)]


async def test_missing_match_does_not_modify_world(tmp_path) -> None:
    world = WorldModel()
    await world.observe_tile(TileObservation(1, 2), cycle=3)
    before = world.snapshot_state()
    store = WorldPersistence(tmp_path / "world.sqlite3", match_id="missing", world=world)

    assert await store.restore() is False
    assert world.snapshot_state() == before


async def test_snapshot_io_does_not_block_event_loop(tmp_path, monkeypatch) -> None:
    world = await _populated_world()
    store = WorldPersistence(tmp_path / "world.sqlite3", match_id="match", world=world)
    original_write = persistence_module._write_snapshot

    def slow_write(*args, **kwargs) -> None:
        time.sleep(0.2)
        original_write(*args, **kwargs)

    monkeypatch.setattr(persistence_module, "_write_snapshot", slow_write)
    snapshot_task = asyncio.create_task(store.snapshot())
    await asyncio.sleep(0)
    await asyncio.sleep(0.03)

    assert not snapshot_task.done()
    await snapshot_task


async def test_cadence_task_and_final_flush(tmp_path) -> None:
    database = tmp_path / "world.sqlite3"
    world = WorldModel()
    store = WorldPersistence(
        database, match_id="match", world=world, interval_s=0.01
    )
    task = await store.start()
    await world.observe_tile(TileObservation(4, 5, terrain_type=2), cycle=11)
    await asyncio.sleep(0.04)
    await store.stop()

    assert task.done()
    restored = WorldModel()
    assert await WorldPersistence(database, match_id="match", world=restored).restore()
    assert restored.get_tile((4, 5)) is not None
    assert restored.cycle == 11


async def test_sqlite_uses_wal_journal_mode(tmp_path) -> None:
    database = tmp_path / "world.sqlite3"
    store = WorldPersistence(database, match_id="match", world=WorldModel())
    await store.snapshot()

    with sqlite3.connect(database) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert journal_mode == "wal"


async def test_invalid_schema_is_rejected(tmp_path) -> None:
    database = tmp_path / "world.sqlite3"
    store = WorldPersistence(database, match_id="match", world=WorldModel())
    await store.snapshot()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE world_snapshots SET schema_version = 999 WHERE match_id = 'match'"
        )

    with pytest.raises(persistence_module.SnapshotError, match="unsupported"):
        await WorldPersistence(database, match_id="match", world=WorldModel()).restore()


async def test_snapshot_state_is_detached_from_live_model() -> None:
    world = WorldModel()
    await world.observe_tile(TileObservation(0, 0, terrain_type=1), cycle=1)
    state = world.snapshot_state()
    await world.observe_tile(TileObservation(0, 0, terrain_type=3), cycle=2)

    assert state.tiles[0].terrain_type == 1
    assert world.get_tile((0, 0)).terrain_type == 3
