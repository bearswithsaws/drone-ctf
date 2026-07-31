"""Startup reconciliation tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.resync import StartupResync


@dataclass
class FakeResult:
    data: dict[str, Any]
    ok: bool = True
    status: int = 200


class FakeRest:
    def __init__(self) -> None:
        self.responses: dict[str, FakeResult | BaseException] = {
            "/queue/drones": FakeResult({"actions": []}),
            "/queue/buildings": FakeResult({"actions": []}),
        }
        self.posts: list[tuple[str, dict[str, Any] | None]] = []

    async def get(self, path: str, params=None) -> FakeResult:
        result = self.responses[path]
        if isinstance(result, BaseException):
            raise result
        return result

    async def post(self, path: str, json=None) -> FakeResult:
        self.posts.append((path, json))
        return FakeResult({"action_id": "status-1"})


async def test_resync_gap_fills_reads_queues_and_requests_one_report() -> None:
    rest = FakeRest()
    rest.responses["/queue/drones"] = FakeResult(
        {"actions": [{"action_id": "d1"}, {"action_id": "d2"}]}
    )
    rest.responses["/queue/buildings"] = FakeResult(
        {"actions": {"base-1": [{"action_id": "b1"}]}}
    )
    gap_calls = 0

    async def gap_fill() -> int:
        nonlocal gap_calls
        gap_calls += 1
        return 4

    resync = StartupResync(rest, gap_fill=gap_fill)
    result = await resync.run()

    assert gap_calls == 1
    assert result.complete
    assert result.gap_filled == 4
    assert result.drone_queue_depth == 2
    assert result.building_queue_depth == 1
    assert result.status_action_id == "status-1"
    assert rest.posts == [
        ("/buildings/command_centre/status_report", {"efficiency": 1})
    ]


async def test_pending_status_report_is_not_duplicated() -> None:
    rest = FakeRest()
    rest.responses["/queue/buildings"] = FakeResult(
        {
            "actions": [
                {
                    "action_id": "already-there",
                    "subject": "Building status_report queued",
                }
            ]
        }
    )

    result = await StartupResync(rest).run()

    assert result.complete
    assert result.status_already_pending
    assert result.status_action_id is None
    assert rest.posts == []


async def test_resync_is_degraded_not_fatal_when_server_calls_fail() -> None:
    rest = FakeRest()
    rest.responses["/queue/drones"] = RuntimeError("offline")

    async def failed_gap_fill() -> int:
        raise RuntimeError("inbox unavailable")

    result = await StartupResync(rest, gap_fill=failed_gap_fill).run()

    assert not result.complete
    assert result.drone_queue_depth is None
    assert result.building_queue_depth == 0
    assert len(result.errors) == 2
    # The independent status sweep still runs despite the other failures.
    assert result.status_action_id == "status-1"
