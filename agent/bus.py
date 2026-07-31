"""Async publish/subscribe event bus.

Decouples producers (transport) from consumers (action tracker, world ingest,
telemetry) so the seed client's giant ``elif`` message dispatcher can be
replaced by independent subscribers. Handlers are async callables; one failing
handler never prevents the others from running.

Subscribe to a specific topic, or to ``"*"`` to receive every published event.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable

log = logging.getLogger("agent.bus")

Handler = Callable[[str, Any], Awaitable[None]]

WILDCARD = "*"


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Handler) -> Callable[[], None]:
        """Register ``handler`` for ``topic`` (or ``"*"``). Returns an unsubscribe fn."""
        self._subs[topic].append(handler)

        def _unsub() -> None:
            try:
                self._subs[topic].remove(handler)
            except ValueError:
                pass

        return _unsub

    async def publish(self, topic: str, payload: Any) -> None:
        """Deliver ``payload`` to all handlers of ``topic`` and of ``"*"``.

        Handlers run concurrently; exceptions are logged and isolated.
        """
        handlers = list(self._subs.get(topic, ())) + list(self._subs.get(WILDCARD, ()))
        if not handlers:
            return
        results = await asyncio.gather(
            *(h(topic, payload) for h in handlers), return_exceptions=True
        )
        for h, res in zip(handlers, results):
            if isinstance(res, Exception):
                log.error("Handler %r failed on topic %s: %r", getattr(h, "__name__", h), topic, res)
