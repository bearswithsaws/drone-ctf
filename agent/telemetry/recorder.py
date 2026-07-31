"""Append-only JSON Lines recording for deterministic session replay.

Each line is a self-contained envelope.  The future E3.6 replay harness can
select ``topic == "message"`` and feed ``payload`` through world ingest while
retaining submitted actions for decision auditing.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import IO, Any

from agent.bus import EventBus
from agent.transport.action_tracker import TOPIC_ACTION_SUBMITTED
from agent.transport.ws import TOPIC_MESSAGE

RECORDING_VERSION = 1
RECORDED_TOPICS = frozenset({TOPIC_MESSAGE, TOPIC_ACTION_SUBMITTED})


def _json_default(value: Any) -> Any:
    """Convert the typed objects used on the event bus to JSON values."""

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


class JsonlRecorder:
    """Record inbound messages and submitted actions to an append-only file.

    Constructing with a bus attaches the recorder immediately.  Writes are
    flushed per event, serialized by an asyncio lock, and the file is always
    opened in append mode so prior sessions are never truncated.
    """

    def __init__(
        self,
        path: str | Path,
        bus: EventBus | None = None,
        *,
        clock: Callable[[], float] = time.time,
        session_id: str | None = None,
        fsync: bool = False,
    ) -> None:
        self.path = Path(path)
        self._clock = clock
        self.session_id = session_id or uuid.uuid4().hex
        self._fsync = fsync
        self._file: IO[str] | None = None
        self._lock = asyncio.Lock()
        self._sequence = 0
        self._unsubscribers: list[Callable[[], None]] = []
        if bus is not None:
            self.attach(bus)

    @property
    def records_written(self) -> int:
        return self._sequence

    def attach(self, bus: EventBus) -> None:
        """Subscribe to recordable topics on ``bus``.

        Calling ``attach`` more than once intentionally attaches once per call;
        this keeps ownership explicit when multiple buses are in use.
        """

        for topic in RECORDED_TOPICS:
            self._unsubscribers.append(bus.subscribe(topic, self._on_event))

    async def _on_event(self, topic: str, payload: Any) -> None:
        await self.record(topic, payload)

    async def record(self, topic: str, payload: Any) -> None:
        """Append one replay envelope.

        ``record`` is public so importers can record events without an
        :class:`EventBus`, but only the two stable replay topics are accepted.
        """

        if topic not in RECORDED_TOPICS:
            raise ValueError(f"unsupported recording topic: {topic!r}")

        async with self._lock:
            if self._file is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._file = self.path.open("a", encoding="utf-8")
            self._sequence += 1
            envelope = {
                "version": RECORDING_VERSION,
                "session_id": self.session_id,
                "sequence": self._sequence,
                "recorded_at": self._clock(),
                "topic": topic,
                "payload": payload,
            }
            line = json.dumps(
                envelope,
                default=_json_default,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            self._file.write(line + "\n")
            self._file.flush()
            if self._fsync:
                os.fsync(self._file.fileno())

    async def close(self) -> None:
        """Detach from event buses and close the recording file."""

        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        async with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    async def __aenter__(self) -> JsonlRecorder:  # noqa: PYI034
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()


# Readable compatibility names for callers that do not care about the format.
TelemetryRecorder = JsonlRecorder
Recorder = JsonlRecorder
