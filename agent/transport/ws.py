"""Socket.IO consumer for the game server's real-time message stream.

Ports the seed client's handshake (connect → emit ``authenticate {token}`` →
``authenticated`` → stream of ``message`` events, see ``message_parser.py``)
onto an async client, and adds two things the seed client lacks:

- **Explicit exponential-backoff reconnect** — we manage the connect loop
  ourselves (socketio's built-in auto-reconnect is disabled) so reconnection and
  gap-fill happen in a controlled order.
- **Gap-fill on (re)connect** — after every successful connect we pull the
  server's recent-message backlog via ``GET /messages`` (non-consuming) and
  replay anything we haven't already seen. The message inbox is capped at the
  last 100 per user, so a disconnect longer than 100 messages can still lose
  data; that window is recorded, not silently swallowed.

Every inbound message is de-duplicated by ``message_id`` and published to the
event bus under the :data:`TOPIC_MESSAGE` topic. Downstream consumers (action
tracker, world ingest, telemetry) subscribe there.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import Any, Awaitable, Callable

import socketio

from agent.bus import EventBus

log = logging.getLogger("agent.transport.ws")

TOPIC_MESSAGE = "message"

TokenProvider = Callable[[], str | None]


class GameSocket:
    def __init__(
        self,
        server_url: str,
        token_provider: TokenProvider,
        bus: EventBus,
        *,
        gap_fill: Callable[[int], Awaitable[list[dict[str, Any]]]] | None = None,
        socketio_path: str = "socket.io",
        initial_backoff_s: float = 1.0,
        max_backoff_s: float = 30.0,
        gap_fill_count: int = 100,
        seen_cache_size: int = 4000,
        sio: socketio.AsyncClient | None = None,
    ) -> None:
        self._server_url = server_url.rstrip("/")
        self._token_provider = token_provider
        self._bus = bus
        self._gap_fill = gap_fill
        self._socketio_path = socketio_path
        self._initial_backoff = initial_backoff_s
        self._max_backoff = max_backoff_s
        self._gap_fill_count = gap_fill_count

        # We drive reconnection ourselves for deterministic gap-fill ordering.
        self._sio = sio or socketio.AsyncClient(reconnection=False, logger=False, engineio_logger=False)
        self._authenticated = False
        self._stop = False

        # Bounded de-dup cache: message_id -> None, insertion-ordered.
        self._seen: "OrderedDict[str, None]" = OrderedDict()
        self._seen_cap = seen_cache_size

        self._register_handlers()

    # --- public API ---

    @property
    def is_connected(self) -> bool:
        return self._sio.connected and self._authenticated

    async def run(self) -> None:
        """Connect-and-listen loop with exponential backoff. Runs until stop()."""
        backoff = self._initial_backoff
        while not self._stop:
            try:
                token_present = self._token_provider() is not None
                if not token_present:
                    log.warning("No token available yet; delaying socket connect.")
                    await asyncio.sleep(self._initial_backoff)
                    continue
                await self._sio.connect(
                    self._server_url,
                    socketio_path=self._socketio_path,
                    transports=["websocket", "polling"],
                )
                backoff = self._initial_backoff  # reset on a clean connect
                # Handshake (authenticate) runs in the connect handler. Gap-fill
                # after the socket is up so we recover anything missed while down.
                await self._do_gap_fill()
                await self._sio.wait()  # blocks until disconnected
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # connection/transport errors → back off and retry
                log.warning("Socket connect/listen error (%s); retry in %.1fs", exc, backoff)
            finally:
                self._authenticated = False

            if self._stop:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._max_backoff)

    async def stop(self) -> None:
        self._stop = True
        if self._sio.connected:
            await self._sio.disconnect()

    # --- handlers ---

    def _register_handlers(self) -> None:
        @self._sio.event
        async def connect() -> None:  # noqa: unused — registered by decorator
            log.info("Socket connected; authenticating...")
            await self._sio.emit("authenticate", {"token": self._token_provider()})

        @self._sio.on("connected")
        async def on_connected(data: Any) -> None:
            log.debug("Server 'connected': %s", data)

        @self._sio.on("authenticated")
        async def on_authenticated(data: Any) -> None:
            uid = data.get("user_id") if isinstance(data, dict) else None
            self._authenticated = True
            log.info("Socket authenticated (user_id=%s)", uid)

        @self._sio.on("message")
        async def on_message(data: Any) -> None:
            await self._ingest(data, backfilled=False)

        @self._sio.on("error")
        async def on_error(data: Any) -> None:
            msg = data.get("message") if isinstance(data, dict) else data
            log.error("Socket error: %s", msg)

        @self._sio.event
        async def disconnect() -> None:  # noqa: unused
            self._authenticated = False
            log.warning("Socket disconnected")

    # --- ingest / dedup / gap-fill ---

    async def _ingest(self, msg: Any, *, backfilled: bool) -> None:
        """De-dup by message_id and publish to the bus. Messages without an id
        always pass through (can't be de-duped, better to double-deliver than drop)."""
        if not isinstance(msg, dict):
            log.warning("Ignoring non-dict socket message: %r", msg)
            return
        mid = msg.get("message_id")
        if mid is not None:
            if mid in self._seen:
                return
            self._seen[mid] = None
            if len(self._seen) > self._seen_cap:
                self._seen.popitem(last=False)  # evict oldest
        if backfilled:
            msg = {**msg, "_backfilled": True}
        await self._bus.publish(TOPIC_MESSAGE, msg)

    async def _do_gap_fill(self) -> None:
        if self._gap_fill is None:
            return
        try:
            messages = await self._gap_fill(self._gap_fill_count)
        except Exception as exc:
            log.warning("Gap-fill request failed: %s", exc)
            return
        new = 0
        for m in messages:
            before = len(self._seen)
            await self._ingest(m, backfilled=True)
            if len(self._seen) != before:
                new += 1
        if new:
            log.info("Gap-fill replayed %d missed message(s) of %d fetched", new, len(messages))
        # The inbox caps at 100; a full page means we may have lost older ones.
        if len(messages) >= self._gap_fill_count:
            log.warning(
                "Gap-fill returned a full page (%d); messages older than the "
                "server's 100-message cap may have been lost during downtime.",
                len(messages),
            )
