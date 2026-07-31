"""Network-free integration coverage for the production runtime composition."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.config import Config
from agent.replay import replay_recording
from agent.runtime import PlanningContext, compose_runtime
from agent.telemetry.metrics import TOPIC_QUEUE_DEPTH, QueueDepthEvent
from agent.transport.parsing import TOPIC_PARSED_MESSAGE
from agent.transport.schema import UnknownWireMessage


@dataclass
class FakeResult:
    data: dict[str, Any]
    ok: bool = True
    status: int = 200


class FakeRest:
    def __init__(self) -> None:
        self.token = "token"
        self.closed = False

    async def get(self, path: str, params: dict[str, Any] | None = None) -> FakeResult:
        if path in {"/queue/drones", "/queue/buildings"}:
            return FakeResult({"actions": []})
        raise AssertionError(f"unexpected GET {path}")

    async def delete(self, path: str, json: dict[str, Any] | None = None) -> FakeResult:
        return FakeResult({})

    async def post(self, path: str, json: dict[str, Any] | None = None) -> FakeResult:
        return FakeResult({"action_id": "post-action-1"})

    async def aclose(self) -> None:
        self.closed = True


class PublishingSocket:
    def __init__(self, *_args: Any, bus: Any, **_kwargs: Any) -> None:
        self.bus = bus
        self.is_connected = True
        self.published = asyncio.Event()
        self._stop = asyncio.Event()

    async def run(self) -> None:
        for message in LIVE_MESSAGES:
            await self.bus.publish("message", message)
        self.published.set()
        await self._stop.wait()

    async def stop(self) -> None:
        self._stop.set()


class IdleSocket:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.is_connected = True
        self._stop = asyncio.Event()

    async def run(self) -> None:
        await self._stop.wait()

    async def stop(self) -> None:
        self._stop.set()


class RecordingStrategy:
    def __init__(self) -> None:
        self.context: PlanningContext | None = None
        self.stopped = False

    async def start(self, context: PlanningContext) -> None:
        self.context = context

    async def stop(self) -> None:
        self.stopped = True


LIVE_MESSAGES = [
    {
        "message_id": "report-1",
        "message_type": "building_action_completed",
        "action_id": "report-action",
        "building_id": "cc-1",
        "subject": "Building status_report completed",
        "timestamp": 10,
        "details": {
            "success": True,
            "buildings": [
                {
                    "building_id": "cc-1",
                    "building_type": "command_center",
                    "origin": [0, 0],
                    "coordinates": [[0, 0]],
                }
            ],
            "drones": [{"drone_id": "scout-1", "location": [2, -1]}],
            "summary": {
                "total_drones": 1,
                "active_drones": 1,
                "total_buildings": 1,
                "active_buildings": 1,
                "powered_buildings": 1,
                "command_center_resources": {
                    "titanium_parts": 80,
                    "battery_materials": 35,
                },
                "cycle": 10,
            },
            "drone_count": 1,
            "building_count": 1,
        },
    },
    {
        "message_id": "status-1",
        "message_type": "action_completed",
        "action_id": "status-action",
        "drone_id": "scout-1",
        "subject": "Status completed",
        "timestamp": 11,
        "details": {
            "success": True,
            "status": {
                "drone_id": "scout-1",
                "current_battery": 100,
                "max_battery": 120,
                "speed": 0,
                "max_health": 100,
                "current_health": 100,
                "armour": 0,
                "equipment_slots": 4,
                "chassis_weight": 20,
                "direction": 2,
                "elevation": 1,
                "equipment": [],
                "cargo": {},
                "shields": 0,
                "max_shields": 0,
                "base_shields": 0,
                "upgrade_tier_applied": 0,
                "last_action_cycle": 11,
                "is_decoy": False,
                "is_lucas": False,
                "health_percentage": 100,
                "available_equipment_slots": 4,
            },
        },
    },
    {
        "message_id": "scan-1",
        "message_type": "action_completed",
        "action_id": "scan-action",
        "drone_id": "scout-1",
        "subject": "Scan completed",
        "timestamp": 12,
        "details": {
            "success": True,
            "scan_range": 1,
            "tiles_scanned": 1,
            "scan_data": {
                "tiles": [
                    {
                        "coordinates": [0, 0],
                        "distance": 0,
                        "terrain": "clear",
                        "terrain_type": 1,
                        "elevation": 1,
                        "has_resource": False,
                        "has_building": False,
                        "has_drone": True,
                    }
                ],
                "summary": {
                    "total_tiles": 1,
                    "tiles_with_resources": 0,
                    "tiles_with_drones": 1,
                    "tiles_with_buildings": 0,
                },
            },
        },
    },
]


def config(tmp_path: Path, *, recording: str, planning_enabled: bool = False) -> Config:
    return Config(
        "https://game.test",
        "pilot",
        "secret",
        telemetry_path=tmp_path / recording,
        world_database=tmp_path / "world.sqlite",
        match_id="match-77",
        snapshot_interval_s=60,
        planning_enabled=planning_enabled,
    )


async def test_startup_event_flow_restart_and_graceful_shutdown(tmp_path: Path) -> None:
    first_rest = FakeRest()
    first = compose_runtime(
        config(tmp_path, recording="session-1.jsonl"),
        first_rest,
        socket_factory=PublishingSocket,
    )
    parsed: list[Any] = []

    async def capture_parsed(_topic: str, message: Any) -> None:
        parsed.append(message)

    first.bus.subscribe(TOPIC_PARSED_MESSAGE, capture_parsed)

    await first.start()
    await asyncio.wait_for(first.socket.published.wait(), timeout=1)
    await first.bus.publish(TOPIC_QUEUE_DEPTH, QueueDepthEvent("drone", "scout-1", 3))

    drone = first.world.get_drone("scout-1")
    simulated = first.entity_sim.get_drone("scout-1")
    assert drone is not None and drone.coord == (2, -1) and drone.direction == 2
    assert first.world.get_tile((2, -1)) is not None
    assert simulated is not None and simulated.current_battery == 99
    assert first.metrics.queue_depths == {"drone": {"scout-1": 3}}
    assert len(parsed) == len(LIVE_MESSAGES)
    assert not any(isinstance(message, UnknownWireMessage) for message in parsed)
    tasks = first.background_tasks

    await first.stop()

    assert first_rest.closed
    assert tasks and all(task.done() for task in tasks)
    assert first.persistence.last_snapshot is not None

    replayed = await replay_recording(tmp_path / "session-1.jsonl")
    assert replayed.messages_replayed == len(LIVE_MESSAGES)
    assert replayed.snapshot["world"]["drones"][0]["drone_id"] == "scout-1"
    assert replayed.snapshot["entity_sim"]["drones"][0]["current_battery"] == 99

    second_rest = FakeRest()
    second = compose_runtime(
        config(tmp_path, recording="session-2.jsonl"),
        second_rest,
        socket_factory=IdleSocket,
    )
    await second.start()

    restored = second.world.get_drone("scout-1")
    assert second.restored_snapshot
    assert restored is not None and restored.coord == (2, -1)
    restart_tasks = second.background_tasks

    await second.stop()

    assert second_rest.closed
    assert restart_tasks and all(task.done() for task in restart_tasks)


async def test_strategy_boundary_wires_allocator_and_pipeliner_without_controllers(
    tmp_path: Path,
) -> None:
    strategy = RecordingStrategy()
    runtime = compose_runtime(
        config(tmp_path, recording="planning.jsonl"),
        FakeRest(),
        socket_factory=IdleSocket,
        strategy=strategy,
    )

    await runtime.start()

    assert runtime.planning.enabled
    assert runtime.planning.task is not None
    assert strategy.context is not None
    assert strategy.context.allocator is runtime.allocator
    assert strategy.context.pipeliner is runtime.pipeliner

    task = runtime.planning.task
    await runtime.stop()

    assert strategy.stopped
    assert task is not None and task.done()


async def test_tracker_events_feed_runtime_telemetry(tmp_path: Path) -> None:
    runtime = compose_runtime(
        config(tmp_path, recording="tracker.jsonl"),
        FakeRest(),
        socket_factory=IdleSocket,
    )

    await runtime.tracker.submit(
        "local-scan-1",
        "drone",
        "scout-1",
        "sensors/scan",
        precondition=lambda: True,
        payload={"level": 1},
        distance=7,
    )

    assert runtime.metrics.loss_rate_by_distance["5-10"].submissions == 1
    await runtime.stop()

    records = [
        json.loads(line)
        for line in (tmp_path / "tracker.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["topic"] for record in records] == ["action.submitted"]
    assert records[0]["payload"]["local_id"] == "local-scan-1"
