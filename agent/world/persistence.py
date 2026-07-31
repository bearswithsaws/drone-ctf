"""Crash-safe, write-behind SQLite persistence for the world model.

The live model remains entirely in memory.  At each cadence the event-loop
thread captures a small, consistent :class:`~agent.world.model.WorldState`;
JSON encoding and all SQLite work happen in a worker thread.  A snapshot is a
single UPSERT keyed by match id, so a process killed during a write restores
either the previous complete checkpoint or the new complete checkpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.world.model import (
    BuildingRecord,
    DroneRecord,
    EnemyBuildingRecord,
    WorldModel,
    WorldState,
)
from agent.world.tiles import Source, TileBelief

log = logging.getLogger("agent.world.persistence")

SCHEMA_VERSION = 1


class SnapshotError(RuntimeError):
    """A stored snapshot exists but cannot safely be restored."""


@dataclass(frozen=True, slots=True)
class SnapshotMetadata:
    match_id: str
    cycle: int
    saved_at: float
    schema_version: int = SCHEMA_VERSION


class WorldPersistence:
    """Own the snapshot lifecycle for one world model and match.

    Call :meth:`restore` before :meth:`start` during startup.  :meth:`stop`
    performs a final flush by default, while abrupt termination relies on the
    most recent WAL-backed periodic snapshot.
    """

    def __init__(
        self,
        database: str | Path,
        *,
        match_id: str,
        world: WorldModel,
        interval_s: float = 10.0,
    ) -> None:
        if not match_id:
            raise ValueError("match_id must not be empty")
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.database = Path(database)
        self.match_id = match_id
        self.world = world
        self.interval_s = interval_s
        self.last_snapshot: SnapshotMetadata | None = None
        self._io_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Create the database/schema and enable WAL without blocking asyncio."""
        async with self._io_lock:
            await self._initialize_locked()

    async def snapshot(self) -> SnapshotMetadata:
        """Persist one complete checkpoint, serializing all disk I/O."""
        state = self.world.snapshot_state()
        saved_at = time.time()
        metadata = SnapshotMetadata(self.match_id, state.cycle, saved_at)
        async with self._io_lock:
            await self._initialize_locked()
            await _run_in_worker(
                _write_snapshot,
                self.database,
                metadata,
                state,
            )
        self.last_snapshot = metadata
        return metadata

    async def restore(self) -> bool:
        """Restore this match's latest checkpoint; return ``False`` if absent."""
        async with self._io_lock:
            await self._initialize_locked()
            row = await _run_in_worker(_read_snapshot, self.database, self.match_id)
        if row is None:
            return False

        metadata, state = row
        await self.world.restore_state(state)
        self.last_snapshot = metadata
        return True

    async def run(self) -> None:
        """Snapshot at the configured cadence until :meth:`stop` is requested."""
        await self.initialize()
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                try:
                    await self.snapshot()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("world snapshot failed for match %s", self.match_id)

    async def start(self) -> asyncio.Task[None]:
        """Start the single writer task, returning the existing task if running."""
        if self._task is None or self._task.done():
            self._stop_event.clear()
            self._task = asyncio.create_task(
                self.run(), name=f"world-snapshot:{self.match_id}"
            )
        return self._task

    async def stop(self, *, flush: bool = True) -> None:
        """Stop the cadence loop and optionally persist the current state."""
        self._stop_event.set()
        task = self._task
        if task is not None:
            await task
            self._task = None
        if flush:
            await self.snapshot()

    async def _initialize_locked(self) -> None:
        if not self._initialized:
            await _run_in_worker(_initialize_database, self.database)
            self._initialized = True


def _connect(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database, timeout=30.0)
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _initialize_database(database: Path) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(database)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS world_snapshots (
                    match_id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    snapshot_cycle INTEGER NOT NULL,
                    snapshot_time REAL NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )


def _write_snapshot(
    database: Path,
    metadata: SnapshotMetadata,
    state: WorldState,
) -> None:
    payload = json.dumps(_state_to_payload(state), separators=(",", ":"), sort_keys=True)
    with closing(_connect(database)) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO world_snapshots (
                    match_id, schema_version, snapshot_cycle, snapshot_time, payload
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(match_id) DO UPDATE SET
                    schema_version=excluded.schema_version,
                    snapshot_cycle=excluded.snapshot_cycle,
                    snapshot_time=excluded.snapshot_time,
                    payload=excluded.payload
                """,
                (
                    metadata.match_id,
                    metadata.schema_version,
                    metadata.cycle,
                    metadata.saved_at,
                    payload,
                ),
            )


def _read_snapshot(
    database: Path, match_id: str
) -> tuple[SnapshotMetadata, WorldState] | None:
    with closing(_connect(database)) as connection:
        row = connection.execute(
            """
            SELECT schema_version, snapshot_cycle, snapshot_time, payload
            FROM world_snapshots
            WHERE match_id = ?
            """,
            (match_id,),
        ).fetchone()
    if row is None:
        return None

    version, cycle, saved_at, raw_payload = row
    if version != SCHEMA_VERSION:
        raise SnapshotError(
            f"unsupported world snapshot schema {version}; expected {SCHEMA_VERSION}"
        )
    try:
        payload = json.loads(raw_payload)
        state = _payload_to_state(payload)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SnapshotError(f"invalid world snapshot for match {match_id!r}") from exc
    if state.cycle != cycle:
        raise SnapshotError(
            f"world snapshot cycle mismatch for match {match_id!r}: {cycle} != {state.cycle}"
        )
    return SnapshotMetadata(match_id, cycle, saved_at, version), state


def _state_to_payload(state: WorldState) -> dict[str, Any]:
    return {
        "cycle": state.cycle,
        "confidence_half_life": state.confidence_half_life,
        "tiles": [asdict(tile) for tile in state.tiles],
        "drones": [asdict(drone) for drone in state.drones],
        "buildings": [asdict(building) for building in state.buildings],
        "enemy_buildings": [asdict(building) for building in state.enemy_buildings],
    }


def _payload_to_state(payload: dict[str, Any]) -> WorldState:
    if not isinstance(payload, dict):
        raise TypeError("snapshot payload must be an object")

    tiles = []
    for raw in payload["tiles"]:
        raw = dict(raw)
        raw["source"] = Source(raw["source"])
        tiles.append(TileBelief(**raw))

    return WorldState(
        cycle=int(payload["cycle"]),
        confidence_half_life=int(payload["confidence_half_life"]),
        tiles=tuple(tiles),
        drones=tuple(DroneRecord(**raw) for raw in payload["drones"]),
        buildings=tuple(
            BuildingRecord(
                **{
                    **raw,
                    "origin": _coord_or_none(raw.get("origin")),
                    "tiles": tuple(tuple(coord) for coord in raw.get("tiles", ())),
                }
            )
            for raw in payload["buildings"]
        ),
        enemy_buildings=tuple(
            EnemyBuildingRecord(
                **{
                    **raw,
                    "origin": _coord_or_none(raw.get("origin")),
                    "tiles": tuple(tuple(coord) for coord in raw.get("tiles", ())),
                }
            )
            for raw in payload["enemy_buildings"]
        ),
    )


def _coord_or_none(raw: Any) -> tuple[int, int] | None:
    if raw is None:
        return None
    q, r = raw
    return (int(q), int(r))


async def _run_in_worker(function: Any, *args: Any) -> Any:
    """Run blocking work on a short-lived daemon thread.

    A private worker keeps persistence independent of asyncio's process-wide
    default executor.  The database lock ensures only one worker per store can
    perform SQLite I/O at a time.
    """
    results: list[Any] = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            result = function(*args)
        except BaseException as exc:
            errors.append(exc)
        else:
            results.append(result)

    worker = threading.Thread(
        target=work,
        name="world-persistence-io",
        daemon=True,
    )
    worker.start()
    while worker.is_alive():
        await asyncio.sleep(0.001)
    worker.join()

    if errors:
        raise errors[0]
    return results[0]
