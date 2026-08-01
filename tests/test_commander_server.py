"""Loopback-only coverage for the commander API served by the live runtime."""

from __future__ import annotations

import asyncio
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
import socketio

from agent.commander.server import CommanderServer
from agent.config import Config
from agent.runtime import AgentRuntime, PlanningContext, compose_runtime


class FakeRest:
    def __init__(self) -> None:
        self.token = "token"
        self.closed = False

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        raise AssertionError(f"unexpected GET {path}")

    async def aclose(self) -> None:
        self.closed = True


class IdleSocket:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.is_connected = True
        self.stopped = False
        self._stop = asyncio.Event()

    async def run(self) -> None:
        await self._stop.wait()

    async def stop(self) -> None:
        self.stopped = True
        self._stop.set()


class RecordingStrategy:
    def __init__(self) -> None:
        self.context: PlanningContext | None = None

    async def start(self, context: PlanningContext) -> None:
        self.context = context

    async def stop(self) -> None:
        return None


def config(
    tmp_path: Path,
    *,
    port: int = 0,
    enabled: bool = True,
    cors_origins: tuple[str, ...] = ("http://localhost:5173",),
) -> Config:
    return Config(
        "https://game.test",
        "pilot",
        "secret",
        telemetry_path=tmp_path / "commander.jsonl",
        world_database=tmp_path / "world.sqlite",
        match_id="match-commander",
        snapshot_interval_s=60,
        commander_enabled=enabled,
        commander_host="127.0.0.1",
        commander_port=port,
        commander_cors_origins=cors_origins,
    )


def directive(directive_id: str, stance: str = "aggressive") -> dict[str, Any]:
    return {
        "kind": "stance",
        "directive_id": directive_id,
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "ttl_seconds": 300,
        "stance": stance,
    }


def test_composition_shares_the_runtime_stores_and_directive_store(tmp_path: Path) -> None:
    strategy = RecordingStrategy()
    runtime = compose_runtime(
        config(tmp_path),
        FakeRest(),
        socket_factory=IdleSocket,
        strategy=strategy,
    )

    commander = runtime.commander
    assert commander is not None
    assert commander.world is runtime.world
    assert commander.tracks is runtime.tracks
    assert commander.bus is runtime.bus
    # The planner and the control-plane must read and write one directive store.
    assert commander.directives is runtime.directives
    assert runtime.planning.context.directives is runtime.directives
    assert commander.cors_allowed_origins == ["http://localhost:5173"]
    assert runtime.commander_server is not None
    assert runtime.commander_server.host == "127.0.0.1"


def test_composition_omits_the_commander_when_disabled(tmp_path: Path) -> None:
    runtime = compose_runtime(
        config(tmp_path, enabled=False), FakeRest(), socket_factory=IdleSocket
    )

    assert runtime.commander is None
    assert runtime.commander_server is None


async def test_posted_directive_reaches_the_store_the_planner_reads(tmp_path: Path) -> None:
    strategy = RecordingStrategy()
    runtime = compose_runtime(
        config(tmp_path),
        FakeRest(),
        socket_factory=IdleSocket,
        strategy=strategy,
    )
    commander = runtime.commander
    assert commander is not None

    transport = httpx.ASGITransport(app=commander.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/directives", json=directive("stance-1"))

    assert response.status_code == 202
    state = runtime.planning.context.directives.state
    assert state.stance is not None and state.stance.stance == "aggressive"

    await runtime.stop()


async def test_control_plane_seeds_state_then_receives_a_world_diff(tmp_path: Path) -> None:
    runtime = compose_runtime(config(tmp_path), FakeRest(), socket_factory=IdleSocket)
    await runtime.start()
    server = runtime.commander_server
    assert server is not None and server.is_running
    assert server.bound_port != 0

    diffs: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    client = socketio.AsyncClient()
    client.on("world_diff", lambda payload: diffs.put_nowait(payload))

    try:
        async with httpx.AsyncClient(base_url=server.url, trust_env=False) as http:
            seeded = await http.get("/v1/state")
            assert seeded.status_code == 200
            snapshot = seeded.json()
            assert snapshot["contract_version"] == "1.0"
            assert snapshot["drones"] == []

            await client.connect(server.url, transports=["websocket"])
            await runtime.world.upsert_drone(
                "scout-1", q=2, r=-1, direction=3, elevation=1, cycle=9
            )
            diff = await asyncio.wait_for(diffs.get(), timeout=5)

            plan = await http.get("/v1/plan")
        assert plan.status_code == 200
        assert plan.json()["contract_version"] == "1.0"
    finally:
        await client.disconnect()

    assert diff["sequence"] == snapshot["sequence"] + 1
    assert diff["cycle"] == 9
    assert diff["changes"][0]["type"] == "drone_upsert"
    assert diff["changes"][0]["drone"]["drone_id"] == "scout-1"

    task = server.task
    await runtime.stop()

    assert task is not None and task.done()
    assert not server.is_running
    assert all(background.done() for background in runtime.background_tasks)
    assert not [
        pending
        for pending in asyncio.all_tasks()
        if pending.get_name() == "commander-server"
    ]
    with socket.socket() as probe:
        # The listening socket is released, so the port is immediately reusable.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", server.bound_port))

    # Shutdown detached the projection from the bus.
    commander = runtime.commander
    assert commander is not None
    await runtime.world.upsert_drone("scout-2", q=0, r=0, direction=1, elevation=1, cycle=10)
    assert [drone.drone_id for drone in (await commander.state()).drones] == ["scout-1"]


async def test_startup_failure_is_surfaced_and_partial_startup_is_cleaned_up(
    tmp_path: Path,
) -> None:
    with socket.socket() as taken:
        taken.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = taken.getsockname()[1]

        rest = FakeRest()
        runtime = compose_runtime(
            config(tmp_path, port=port), rest, socket_factory=IdleSocket
        )
        with pytest.raises(RuntimeError, match="cannot bind 127.0.0.1"):
            await runtime.start()

    # The transports that did start are stopped and their tasks are joined.
    assert rest.closed
    assert runtime.socket.stopped
    assert runtime.background_tasks == ()
    assert runtime.commander_server is not None and not runtime.commander_server.is_running
    assert not {task.get_name() for task in asyncio.all_tasks()} & {
        "action-tracker",
        "game-socket",
        "commander-server",
    }


async def test_server_start_is_idempotent_and_stop_joins_the_serving_task(
    tmp_path: Path,
) -> None:
    runtime = compose_runtime(config(tmp_path), FakeRest(), socket_factory=IdleSocket)
    commander = runtime.commander
    assert commander is not None
    server = CommanderServer(commander, host="127.0.0.1", port=0)

    await server.start()
    task = server.task
    await server.start()

    assert server.task is task
    assert server.url.startswith("http://127.0.0.1:")

    await server.stop()
    await server.stop()

    assert task is not None and task.done()
    assert server.task is None
    await runtime.stop()


async def test_server_startup_timeout_cleans_up_the_serving_task(tmp_path: Path) -> None:
    class NeverStartingServer:
        def __init__(self, _config: Any) -> None:
            self.started = False
            self.should_exit = False

        async def serve(self, sockets: list[socket.socket] | None = None) -> None:
            while not self.should_exit:
                await asyncio.sleep(0.01)

    runtime = compose_runtime(config(tmp_path), FakeRest(), socket_factory=IdleSocket)
    commander = runtime.commander
    assert commander is not None
    server = CommanderServer(
        commander,
        host="127.0.0.1",
        port=0,
        startup_timeout_s=0.05,
        server_factory=NeverStartingServer,
    )

    with pytest.raises(TimeoutError, match="did not start within"):
        await server.start()

    assert server.task is None
    assert not server.is_running
    await runtime.stop()


def test_runtime_without_a_commander_reports_no_commander_tasks(tmp_path: Path) -> None:
    runtime: AgentRuntime = compose_runtime(
        config(tmp_path, enabled=False), FakeRest(), socket_factory=IdleSocket
    )

    assert runtime.background_tasks == ()
