"""Unit tests for hello-world runtime helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agent import __main__ as agent_main
from agent.__main__ import _extract_drone_ids, _gap_fill, _wait_for_socket
from agent.config import Config


@dataclass
class FakeResult:
    data: dict[str, Any]
    ok: bool = True
    status: int = 200
    error_message: str = ""


class FakeRest:
    def __init__(self, result: FakeResult) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def get(self, path: str, params: dict[str, Any] | None = None) -> FakeResult:
        self.calls.append((path, params))
        return self.result


class FakeSocket:
    is_connected = False


class RuntimeRest:
    instance: "RuntimeRest"

    def __init__(self, _api_base: str) -> None:
        type(self).instance = self
        self.token: str | None = None
        self.closed = False

    def set_reauth(self, _reauth: object) -> None:
        pass

    def set_token(self, token: str) -> None:
        self.token = token

    async def post(self, path: str, json: dict[str, Any] | None = None) -> FakeResult:
        if path == "/auth/login":
            return FakeResult({"token": "token", "user_id": "user-1"})
        if path == "/buildings/command_centre/status_report":
            return FakeResult({"action_id": "report-1"})
        raise AssertionError(f"unexpected POST {path}")

    async def get(self, path: str, params: dict[str, Any] | None = None) -> FakeResult:
        if path == "/drones":
            return FakeResult({"drone_ids": ["drone-1"]})
        if path == "/auth/scoreboard":
            return FakeResult({"scoreboard": [], "team_scores": {}})
        raise AssertionError(f"unexpected GET {path}")

    async def aclose(self) -> None:
        self.closed = True


class RuntimeTracker:
    instance: "RuntimeTracker"

    def __init__(self, _rest: object, _bus: object, **_kwargs: object) -> None:
        type(self).instance = self
        self.stopped = False
        self.closed = False

    async def run(self) -> None:
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self.stopped = True

    async def on_message(self, _message: object) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class RuntimeSocket:
    instance: "RuntimeSocket"

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        type(self).instance = self
        self.is_connected = True
        self.stopped = False

    async def run(self) -> None:
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self.stopped = True


class RuntimeController:
    selected: str | None = None
    duration: float | None = None

    def __init__(self, _tracker: object, drone_id: str) -> None:
        type(self).selected = drone_id
        self.drone_id = drone_id

    async def prepare(self) -> bool:
        return True

    async def run(self, *, duration_s: float | None = None) -> int:
        type(self).duration = duration_s
        return 2


def test_extract_drone_ids_accepts_documented_and_legacy_shapes() -> None:
    assert _extract_drone_ids({"drone_ids": ["one", "two", "one"]}) == ["one", "two"]
    assert _extract_drone_ids({"drones": [{"drone_id": "three"}, {"id": "four"}]}) == [
        "three",
        "four",
    ]
    assert _extract_drone_ids({"drone_ids": "not-a-list"}) == []


async def test_gap_fill_is_non_consuming_and_filters_malformed_messages() -> None:
    rest = FakeRest(FakeResult({"messages": [{"message_id": "one"}, "bad"]}))

    messages = await _gap_fill(rest, 100)  # type: ignore[arg-type]

    assert messages == [{"message_id": "one"}]
    assert rest.calls == [("/messages", {"count": 100, "consume": False})]


async def test_gap_fill_surfaces_http_failure() -> None:
    rest = FakeRest(FakeResult({"error": "down"}, ok=False, status=503))

    with pytest.raises(RuntimeError, match=r"GET /messages failed \(503\)"):
        await _gap_fill(rest, 10)  # type: ignore[arg-type]


async def test_wait_for_socket_times_out() -> None:
    with pytest.raises(TimeoutError, match="Socket.IO authentication"):
        await _wait_for_socket(FakeSocket(), timeout_s=0.001)  # type: ignore[arg-type]


async def test_runtime_wires_transport_controller_and_clean_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(agent_main, "GameRest", RuntimeRest)
    monkeypatch.setattr(agent_main, "ActionTracker", RuntimeTracker)
    monkeypatch.setattr(agent_main, "GameSocket", RuntimeSocket)
    monkeypatch.setattr(agent_main, "HelloWorldController", RuntimeController)
    cfg = Config(
        "https://game.test",
        "pilot",
        "secret",
        telemetry_path=tmp_path / "runtime.jsonl",
        world_database=tmp_path / "world.sqlite",
    )

    assert await agent_main.run(cfg, mode="proof", duration_s=600) == 0

    assert RuntimeController.selected == "drone-1"
    assert RuntimeController.duration == 600
    assert RuntimeTracker.instance.stopped
    assert RuntimeTracker.instance.closed
    assert RuntimeSocket.instance.stopped
    assert RuntimeRest.instance.closed


async def test_play_mode_installs_strategist_and_runs_for_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(agent_main, "GameRest", RuntimeRest)
    monkeypatch.setattr(agent_main, "ActionTracker", RuntimeTracker)
    monkeypatch.setattr(agent_main, "GameSocket", RuntimeSocket)
    cfg = Config(
        "https://game.test",
        "pilot",
        "secret",
        telemetry_path=tmp_path / "runtime.jsonl",
        world_database=tmp_path / "world.sqlite",
    )

    assert await agent_main.run(cfg, mode="play", duration_s=0.05) == 0

    assert RuntimeTracker.instance.stopped
    assert RuntimeSocket.instance.stopped
    assert RuntimeRest.instance.closed
