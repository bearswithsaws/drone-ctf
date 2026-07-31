"""EventBus tests — pure, no network."""

from __future__ import annotations

from agent.bus import EventBus


async def test_publish_to_topic_subscriber() -> None:
    bus = EventBus()
    got: list = []

    async def h(topic: str, payload) -> None:
        got.append((topic, payload))

    bus.subscribe("message", h)
    await bus.publish("message", {"x": 1})
    await bus.publish("other", {"y": 2})  # no subscriber
    assert got == [("message", {"x": 1})]


async def test_wildcard_receives_all() -> None:
    bus = EventBus()
    seen: list = []

    async def star(topic: str, payload) -> None:
        seen.append(topic)

    bus.subscribe("*", star)
    await bus.publish("a", 1)
    await bus.publish("b", 2)
    assert seen == ["a", "b"]


async def test_multiple_handlers_all_run() -> None:
    bus = EventBus()
    calls: list = []

    async def h1(t, p) -> None:
        calls.append("h1")

    async def h2(t, p) -> None:
        calls.append("h2")

    bus.subscribe("t", h1)
    bus.subscribe("t", h2)
    await bus.publish("t", None)
    assert sorted(calls) == ["h1", "h2"]


async def test_handler_exception_isolated() -> None:
    bus = EventBus()
    ok: list = []

    async def boom(t, p) -> None:
        raise RuntimeError("nope")

    async def good(t, p) -> None:
        ok.append(True)

    bus.subscribe("t", boom)
    bus.subscribe("t", good)
    await bus.publish("t", None)  # must not raise
    assert ok == [True]


async def test_unsubscribe() -> None:
    bus = EventBus()
    got: list = []

    async def h(t, p) -> None:
        got.append(p)

    unsub = bus.subscribe("t", h)
    await bus.publish("t", 1)
    unsub()
    await bus.publish("t", 2)
    assert got == [1]
