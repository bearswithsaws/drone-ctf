from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest

from agent.bus import EventBus
from agent.commander.api import CommanderAPI, TOPIC_COMMANDER_ALERT, TOPIC_PLAN_CHANGED
from agent.commander.contract import (
    Alert,
    Coordinate,
    PlanAssignment,
    PlannedPath,
    PlanSnapshot,
    PlanTask,
)
from agent.world.model import TileObservation, WorldModel
from agent.world.tiles import Source
from agent.world.tracks import BearingSighting, Sighting, SightingSource, TrackStore


@pytest.fixture
def commander() -> CommanderAPI:
    bus = EventBus()
    api = CommanderAPI(WorldModel(bus), TrackStore(bus), bus)
    api.sio.emit = AsyncMock()
    return api


async def test_get_state_returns_initial_versioned_snapshot(commander: CommanderAPI) -> None:
    await commander.world.observe_tile(
        TileObservation(q=2, r=-1, terrain_type=1, elevation=4),
        cycle=7,
        source=Source.IDENTIFY,
    )
    await commander.world.upsert_drone(
        "drone-1", q=2, r=-1, direction=3, elevation=4, cycle=7
    )

    transport = httpx.ASGITransport(app=commander.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/state")

    assert response.status_code == 200
    body = response.json()
    assert body["contract_version"] == "1.0"
    assert body["sequence"] == 2
    assert body["cycle"] == 7
    assert body["tiles"] == [
        {
            "coordinate": {"q": 2, "r": -1},
            "terrain_type": 1,
            "elevation": 4,
            "has_resource": False,
            "resource_type": None,
            "resource_amount": None,
            "has_building": False,
            "building_id": None,
            "has_drone": False,
            "last_seen": 7,
            "source": "identify",
            "confidence": 1.0,
        }
    ]
    assert body["drones"][0]["drone_id"] == "drone-1"


async def test_get_plan_returns_initial_empty_snapshot(commander: CommanderAPI) -> None:
    transport = httpx.ASGITransport(app=commander.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/plan")

    assert response.status_code == 200
    assert response.json() == {
        "contract_version": "1.0",
        "sequence": 0,
        "cycle": 0,
        "tasks": [],
        "assignments": [],
        "paths": [],
    }


async def test_get_plan_returns_latest_live_plan(commander: CommanderAPI) -> None:
    first = PlanSnapshot(
        contract_version="1.0",
        sequence=1,
        cycle=19,
        tasks=[],
        assignments=[],
        paths=[],
    )
    latest = PlanSnapshot(
        contract_version="1.0",
        sequence=2,
        cycle=20,
        tasks=[
            PlanTask(
                task_id="mine-iron",
                kind="mine_loop",
                priority=1.5,
                max_assignees=2,
                parameters={"dropoff_id": "refinery-1", "resource_type": "iron"},
            )
        ],
        assignments=[
            PlanAssignment(
                entity_id="drone-1",
                task_id="mine-iron",
                fitness=0.875,
                pinned=False,
                retained=True,
            )
        ],
        paths=[
            PlannedPath(
                entity_id="drone-1",
                coordinates=[Coordinate(q=1, r=2), Coordinate(q=2, r=2)],
            )
        ],
    )

    await commander.bus.publish(TOPIC_PLAN_CHANGED, first)
    await commander.bus.publish(TOPIC_PLAN_CHANGED, latest)

    transport = httpx.ASGITransport(app=commander.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/plan")

    assert response.status_code == 200
    assert response.json() == latest.model_dump(mode="json")


async def test_world_mutations_emit_incremental_diffs(commander: CommanderAPI) -> None:
    initial = await commander.state()

    await commander.world.observe_tile(TileObservation(q=4, r=5), cycle=11)
    await commander.world.upsert_drone("drone-2", q=4, r=5, cycle=11)

    calls = commander.sio.emit.await_args_list
    assert [call.args[0] for call in calls] == ["world_diff", "world_diff"]
    assert calls[0].args[1]["sequence"] == initial.sequence + 1
    assert calls[0].args[1]["changes"][0]["type"] == "tile_upsert"
    assert calls[1].args[1]["sequence"] == initial.sequence + 2
    assert calls[1].args[1]["changes"][0]["type"] == "drone_upsert"

    await commander.world.remove_drone("drone-2")

    removed = commander.sio.emit.await_args_list[-1].args[1]
    assert removed["changes"] == [
        {"type": "entity_removed", "entity_type": "drone", "entity_id": "drone-2"}
    ]


async def test_hostile_track_emits_diff_and_alert(commander: CommanderAPI) -> None:
    track = await commander.tracks.observe(
        Sighting(
            source=SightingSource.IDENTIFY,
            q=8,
            r=3,
            cycle=12,
            drone_id="enemy-1",
            equipment=frozenset({"howitzer"}),
        )
    )

    calls = commander.sio.emit.await_args_list
    assert [call.args[0] for call in calls] == ["world_diff", "alert"]
    diff = calls[0].args[1]
    assert diff["changes"][0]["enemy_track"]["track_id"] == track.track_id
    alert = calls[1].args[1]
    assert alert["severity"] == "critical"
    assert alert["code"] == "enemy_track_detected"
    assert alert["entity_ids"] == ["enemy-1"]
    assert alert["coordinate"] == {"q": 8, "r": 3}


async def test_plan_and_external_alert_topics_are_streamed(commander: CommanderAPI) -> None:
    plan = PlanSnapshot(
        contract_version="1.0",
        sequence=3,
        cycle=20,
        tasks=[],
        assignments=[],
        paths=[],
    )
    alert = Alert(
        alert_id="alert-1",
        severity="warning",
        code="queue_stalled",
        message="Drone queue stalled",
        cycle=20,
        created_at=datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc),
        entity_ids=["drone-1"],
        coordinate=None,
    )

    await commander.bus.publish(TOPIC_PLAN_CHANGED, plan)
    await commander.bus.publish(TOPIC_COMMANDER_ALERT, alert)

    calls = commander.sio.emit.await_args_list
    assert [call.args[0] for call in calls] == ["plan_changed", "alert"]
    assert calls[0].args[1] == plan.model_dump(mode="json")
    assert calls[1].args[1] == alert.model_dump(mode="json")


async def test_enemy_bearing_pushes_threat_alert(commander: CommanderAPI) -> None:
    await commander.tracks.observe_bearing(
        BearingSighting(
            source=SightingSource.SEISMIC,
            sensor_q=1,
            sensor_r=2,
            direction=4,
            cycle=21,
        )
    )

    commander.sio.emit.assert_awaited_once()
    event, alert = commander.sio.emit.await_args.args
    assert event == "alert"
    assert alert["code"] == "enemy_bearing_detected"
    assert alert["coordinate"] == {"q": 1, "r": 2}


async def test_close_detaches_stream_consumers(commander: CommanderAPI) -> None:
    await commander.close()
    await commander.world.observe_tile(TileObservation(q=0, r=0), cycle=1)

    commander.sio.emit.assert_not_awaited()
