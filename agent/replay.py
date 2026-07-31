"""Deterministic replay of telemetry recordings into local agent state."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Any

from agent.sim import EntitySim
from agent.telemetry.recorder import RECORDING_VERSION
from agent.transport.ws import TOPIC_MESSAGE
from agent.world import Ingestor, TrackStore, WorldModel

SNAPSHOT_VERSION = 3


class ReplayFormatError(ValueError):
    """A recording or expected-state file is not valid replay input."""


class ReplayMismatch(AssertionError):
    """The replayed state differs from its checked-in expectation."""


@dataclass(frozen=True, slots=True)
class ReplayResult:
    records_seen: int
    messages_replayed: int
    snapshot: dict[str, Any]


async def replay_recording(path: str | Path) -> ReplayResult:
    """Replay one recorder-produced JSONL file and return its final state."""

    recording_path = Path(path)
    world = WorldModel()
    tracks = TrackStore()
    ingestor = Ingestor(world, tracks)
    sim = EntitySim()
    records_seen = 0
    messages_replayed = 0

    try:
        recording = recording_path.open(encoding="utf-8")
    except OSError as exc:
        raise ReplayFormatError(f"cannot read recording {recording_path}: {exc}") from exc

    with recording:
        for line_number, line in enumerate(recording, start=1):
            if not line.strip():
                continue
            records_seen += 1
            envelope = _json_object(line, recording_path, line_number)
            version = envelope.get("version")
            if version != RECORDING_VERSION:
                raise ReplayFormatError(
                    f"{recording_path}:{line_number}: unsupported recording version "
                    f"{version!r}; expected {RECORDING_VERSION}"
                )
            if envelope.get("topic") != TOPIC_MESSAGE:
                continue
            payload = envelope.get("payload")
            if not isinstance(payload, dict):
                raise ReplayFormatError(
                    f"{recording_path}:{line_number}: message payload must be a JSON object"
                )
            await ingestor.ingest(payload)
            sim.apply_message(payload)
            messages_replayed += 1

    return ReplayResult(
        records_seen=records_seen,
        messages_replayed=messages_replayed,
        snapshot=snapshot_state(world, tracks, sim),
    )


def snapshot_state(world: WorldModel, tracks: TrackStore, sim: EntitySim) -> dict[str, Any]:
    """Build a stable, JSON-native representation of replay-relevant state."""

    tiles = []
    for tile in sorted(world.tiles(), key=lambda item: (item.q, item.r)):
        value = asdict(tile)
        value["source"] = tile.source.name.lower()
        tiles.append(value)

    enemy_tracks = []
    for track in sorted(tracks.tracks(), key=lambda item: item.track_id):
        enemy_tracks.append(
            {
                "track_id": track.track_id,
                "q": track.q,
                "r": track.r,
                "last_seen": track.last_seen,
                "first_seen": track.first_seen,
                "source": track.source.name.lower(),
                "drone_id": track.drone_id,
                "is_decoy": track.is_decoy,
                "equipment": sorted(track.equipment),
                "equipment_known": track.equipment_known,
                "history": list(track.history),
            }
        )

    snapshot = {
        "snapshot_version": SNAPSHOT_VERSION,
        "world": {
            "cycle": world.cycle,
            "tiles": tiles,
            "drones": _sorted_dataclasses(world.drones(), "drone_id"),
            "buildings": _sorted_dataclasses(world.buildings(), "building_id"),
            "enemy_buildings": _sorted_dataclasses(world.enemy_buildings(), "building_id"),
            "enemy_tracks": enemy_tracks,
        },
        "entity_sim": {
            "drones": _sorted_dataclasses(sim.drones.values(), "drone_id"),
            "buildings": _sorted_dataclasses(sim.buildings.values(), "building_id"),
            "research_tiers": sim.research_tiers,
        },
    }
    # Convert tuples and any other JSON-compatible subclasses to the same plain
    # list/dict/scalar types produced when an expected snapshot is loaded.
    return json.loads(json.dumps(snapshot, sort_keys=True))


def load_expected_snapshot(path: str | Path) -> dict[str, Any]:
    expected_path = Path(path)
    try:
        value = json.loads(expected_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReplayFormatError(f"cannot read expected state {expected_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReplayFormatError(
            f"{expected_path}:{exc.lineno}: invalid expected-state JSON: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise ReplayFormatError(f"{expected_path}: expected state must be a JSON object")
    return value


def assert_expected_snapshot(
    actual: dict[str, Any], expected: dict[str, Any], *, expected_path: str | Path
) -> None:
    """Raise with a readable diff when a final replay state regresses."""

    if actual == expected:
        return
    expected_text = json.dumps(expected, indent=2, sort_keys=True).splitlines()
    actual_text = json.dumps(actual, indent=2, sort_keys=True).splitlines()
    diff = "\n".join(
        unified_diff(
            expected_text,
            actual_text,
            fromfile=str(expected_path),
            tofile="replayed state",
            lineterm="",
        )
    )
    raise ReplayMismatch(f"replayed state does not match expected state\n{diff}")


async def replay_and_assert(recording: str | Path, expected: str | Path) -> ReplayResult:
    result = await replay_recording(recording)
    assert_expected_snapshot(
        result.snapshot,
        load_expected_snapshot(expected),
        expected_path=expected,
    )
    return result


def _json_object(line: str, path: Path, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ReplayFormatError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ReplayFormatError(f"{path}:{line_number}: recording line must be a JSON object")
    return value


def _sorted_dataclasses(values: Any, id_field: str) -> list[dict[str, Any]]:
    return [
        _json_native(asdict(value))
        for value in sorted(values, key=lambda item: getattr(item, id_field))
    ]


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_native(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_json_native(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    return value
