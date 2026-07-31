"""GameSocket ingest/dedup/gap-fill tests — network-free.

The socketio client is stubbed; we exercise the message-handling logic
(``_ingest`` and ``_do_gap_fill``) directly, which is where the real behaviour
lives. The live handshake is covered by an @integration smoke test elsewhere.
"""

from __future__ import annotations

from typing import Any

from agent.bus import EventBus
from agent.transport.ws import TOPIC_MESSAGE, GameSocket


class _StubSio:
    """Minimal stand-in for socketio.AsyncClient used only so GameSocket can
    register handlers without a real connection."""

    def __init__(self) -> None:
        self.connected = False

    def event(self, fn):  # decorator passthrough
        return fn

    def on(self, _name):
        def deco(fn):
            return fn

        return deco


def _make(bus: EventBus, gap_fill=None) -> GameSocket:
    return GameSocket(
        "https://game.test",
        token_provider=lambda: "tok",
        bus=bus,
        gap_fill=gap_fill,
        sio=_StubSio(),
    )


async def _collect(bus: EventBus) -> list[dict[str, Any]]:
    got: list[dict[str, Any]] = []

    async def h(_topic: str, payload) -> None:
        got.append(payload)

    bus.subscribe(TOPIC_MESSAGE, h)
    return got


async def test_ingest_publishes_message() -> None:
    bus = EventBus()
    got = await _collect(bus)
    gs = _make(bus)
    await gs._ingest({"message_id": "a", "subject": "hi"}, backfilled=False)
    assert got == [{"message_id": "a", "subject": "hi"}]


async def test_dedup_by_message_id() -> None:
    bus = EventBus()
    got = await _collect(bus)
    gs = _make(bus)
    await gs._ingest({"message_id": "a"}, backfilled=False)
    await gs._ingest({"message_id": "a"}, backfilled=False)  # duplicate
    assert len(got) == 1


async def test_message_without_id_always_delivered() -> None:
    bus = EventBus()
    got = await _collect(bus)
    gs = _make(bus)
    await gs._ingest({"subject": "no id"}, backfilled=False)
    await gs._ingest({"subject": "no id"}, backfilled=False)
    assert len(got) == 2


async def test_non_dict_message_ignored() -> None:
    bus = EventBus()
    got = await _collect(bus)
    gs = _make(bus)
    await gs._ingest("not a dict", backfilled=False)
    assert got == []


async def test_gap_fill_replays_only_unseen_and_marks_backfilled() -> None:
    bus = EventBus()
    got = await _collect(bus)

    async def gap_fill(count: int) -> list[dict[str, Any]]:
        return [{"message_id": "a"}, {"message_id": "b"}]

    gs = _make(bus, gap_fill=gap_fill)
    # 'a' already seen live before the (re)connect.
    await gs._ingest({"message_id": "a"}, backfilled=False)
    got.clear()

    await gs._do_gap_fill()
    # Only 'b' is new; it is flagged as backfilled.
    assert got == [{"message_id": "b", "_backfilled": True}]


async def test_live_message_after_gap_fill_is_deduped() -> None:
    bus = EventBus()
    got = await _collect(bus)

    async def gap_fill(count: int) -> list[dict[str, Any]]:
        return [{"message_id": "b"}]

    gs = _make(bus, gap_fill=gap_fill)
    await gs._do_gap_fill()          # backfills 'b'
    await gs._ingest({"message_id": "b"}, backfilled=False)  # same msg arrives live
    assert len(got) == 1


async def test_gap_fill_failure_is_swallowed() -> None:
    bus = EventBus()
    got = await _collect(bus)

    async def gap_fill(count: int):
        raise RuntimeError("network down")

    gs = _make(bus, gap_fill=gap_fill)
    await gs._do_gap_fill()  # must not raise
    assert got == []


async def test_seen_cache_is_bounded() -> None:
    bus = EventBus()
    gs = _make(bus)
    gs._seen_cap = 10
    for i in range(50):
        await gs._ingest({"message_id": str(i)}, backfilled=False)
    assert len(gs._seen) <= 10
