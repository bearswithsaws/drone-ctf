"""Deterministic startup reconciliation before autonomy is admitted."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

log = logging.getLogger("agent.resync")

GapFill = Callable[[], Awaitable[int]]


class RestResult(Protocol):
    data: dict[str, Any]

    @property
    def ok(self) -> bool: ...


class RestClient(Protocol):
    async def get(self, path: str, params: dict[str, Any] | None = None) -> RestResult: ...

    async def post(self, path: str, json: dict[str, Any] | None = None) -> RestResult: ...


@dataclass(frozen=True, slots=True)
class StartupResyncResult:
    gap_filled: int
    drone_queue_depth: int | None
    building_queue_depth: int | None
    status_action_id: str | None
    status_already_pending: bool
    errors: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.errors


class StartupResync:
    """Recover server-side state that is not part of the world snapshot."""

    STATUS_REPORT_PATH = "/buildings/command_centre/status_report"

    def __init__(self, rest: RestClient, *, gap_fill: GapFill | None = None) -> None:
        self._rest = rest
        self._gap_fill = gap_fill
        self.last_result: StartupResyncResult | None = None
        self.queues: dict[str, Any] = {}

    async def run(self) -> StartupResyncResult:
        """Replay messages, capture both queues, and request an authoritative report.

        Individual REST failures are recorded but do not prevent the runtime
        from reconnecting. The live socket repeats gap-fill after connecting,
        and periodic queue reconciliation continues recovery from transient
        startup failures.
        """

        errors: list[str] = []
        gap_filled = 0
        if self._gap_fill is not None:
            try:
                gap_filled = await self._gap_fill()
            except Exception as exc:
                errors.append(f"message gap-fill failed: {exc}")
                log.warning(errors[-1])

        queue_results = await asyncio.gather(
            self._read_queue("/queue/drones"),
            self._read_queue("/queue/buildings"),
            return_exceptions=True,
        )
        depths: list[int | None] = []
        for path, value in zip(("/queue/drones", "/queue/buildings"), queue_results):
            if isinstance(value, BaseException):
                errors.append(f"{path} read failed: {value}")
                log.warning(errors[-1])
                depths.append(None)
            else:
                actions, depth = value
                self.queues[path] = actions
                depths.append(depth)

        building_actions = self.queues.get("/queue/buildings")
        status_pending = _contains_status_report(building_actions)
        status_action_id: str | None = None
        if not status_pending:
            try:
                response = await self._rest.post(self.STATUS_REPORT_PATH, json={"efficiency": 1})
                if not response.ok:
                    status = getattr(response, "status", "unknown")
                    raise RuntimeError(f"HTTP {status}")
                candidate = response.data.get("action_id")
                if isinstance(candidate, str):
                    status_action_id = candidate
            except Exception as exc:
                errors.append(f"status sweep failed: {exc}")
                log.warning(errors[-1])

        result = StartupResyncResult(
            gap_filled=gap_filled,
            drone_queue_depth=depths[0],
            building_queue_depth=depths[1],
            status_action_id=status_action_id,
            status_already_pending=status_pending,
            errors=tuple(errors),
        )
        self.last_result = result
        log.info(
            "Startup resync: gap=%d drone_queue=%s building_queue=%s status=%s errors=%d",
            result.gap_filled,
            result.drone_queue_depth,
            result.building_queue_depth,
            "pending" if status_pending else status_action_id or "unavailable",
            len(result.errors),
        )
        return result

    async def _read_queue(self, path: str) -> tuple[Any, int]:
        response = await self._rest.get(path)
        if not response.ok:
            status = getattr(response, "status", "unknown")
            raise RuntimeError(f"HTTP {status}")
        actions = response.data.get("actions", [])
        if not isinstance(actions, (list, Mapping)):
            raise RuntimeError("malformed actions payload")
        return actions, _queue_depth(actions)


def _queue_depth(value: Any) -> int:
    if isinstance(value, list):
        return sum(_queue_depth(item) for item in value)
    if isinstance(value, Mapping):
        if isinstance(value.get("action_id", value.get("id")), str):
            return 1
        return sum(_queue_depth(item) for item in value.values())
    return 0


def _contains_status_report(value: Any) -> bool:
    if isinstance(value, str):
        normalised = "".join(
            character for character in value.lower() if character.isalnum()
        )
        return "statusreport" in normalised
    if isinstance(value, Mapping):
        return any(_contains_status_report(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_status_report(item) for item in value)
    return False


__all__ = ["StartupResync", "StartupResyncResult"]
