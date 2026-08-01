"""Uvicorn lifecycle for the commander ASGI app.

The runtime — not this module and not uvicorn — owns the process. The server
therefore binds its listening socket eagerly (so a busy port fails the caller's
``start`` instead of tearing down the event loop), never installs signal
handlers, and joins its serving task on ``stop``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
from collections.abc import Callable, Iterator
from typing import Any

import uvicorn

from agent.commander.api import CommanderAPI

log = logging.getLogger("agent.commander.server")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
STARTUP_TIMEOUT_S = 10.0
SHUTDOWN_TIMEOUT_S = 10.0


class _RuntimeOwnedServer(uvicorn.Server):
    """A uvicorn server that leaves SIGINT/SIGTERM to the agent process."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


class CommanderServer:
    """Serve one :class:`CommanderAPI` over HTTP and Socket.IO."""

    def __init__(
        self,
        commander: CommanderAPI,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        startup_timeout_s: float = STARTUP_TIMEOUT_S,
        shutdown_timeout_s: float = SHUTDOWN_TIMEOUT_S,
        server_factory: Callable[[uvicorn.Config], Any] = _RuntimeOwnedServer,
    ) -> None:
        self.commander = commander
        self.host = host
        self.port = port
        self.startup_timeout_s = startup_timeout_s
        self.shutdown_timeout_s = shutdown_timeout_s
        self._server_factory = server_factory
        self._server: Any | None = None
        self._socket: socket.socket | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def bound_port(self) -> int:
        """The port actually listened on, which resolves an ephemeral ``0``."""

        if self._socket is None:
            return self.port
        return int(self._socket.getsockname()[1])

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.bound_port}"

    async def start(self) -> None:
        """Bind, serve, and return once the app is accepting requests."""

        if self._task is not None:
            return
        self._socket = sock = self._bind()
        server = self._server_factory(
            uvicorn.Config(
                self.commander.app,
                host=self.host,
                port=self.bound_port,
                # Keep the agent's own logging configuration authoritative.
                log_config=None,
                access_log=False,
                lifespan="on",
            )
        )
        self._server = server
        self._task = asyncio.create_task(self._serve(server, sock), name="commander-server")
        try:
            await self._await_started(server)
        except BaseException:
            with contextlib.suppress(BaseException):
                await self.stop()
            raise
        log.info("Commander API serving on %s", self.url)

    async def stop(self) -> None:
        """Ask uvicorn to exit, join its task, and release the socket."""

        server, sock, task = self._server, self._socket, self._task
        self._server = self._socket = self._task = None
        try:
            if server is not None:
                server.should_exit = True
            if task is not None:
                try:
                    await asyncio.wait_for(task, timeout=self.shutdown_timeout_s)
                except asyncio.TimeoutError:
                    log.error(
                        "Commander server did not stop within %gs; cancelled",
                        self.shutdown_timeout_s,
                    )
                except asyncio.CancelledError:
                    if not task.cancelled():
                        raise
        finally:
            if sock is not None:
                # Already closed by uvicorn when its startup completed.
                with contextlib.suppress(OSError):
                    sock.close()

    def _bind(self) -> socket.socket:
        family = socket.AF_INET6 if ":" in self.host else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.host, self.port))
        except OSError as exc:
            sock.close()
            raise RuntimeError(
                f"commander server cannot bind {self.host}:{self.port}: {exc}"
            ) from exc
        sock.set_inheritable(True)
        return sock

    async def _serve(self, server: Any, sock: socket.socket) -> None:
        try:
            await server.serve(sockets=[sock])
        except SystemExit as exc:  # uvicorn aborts a failed startup with sys.exit
            raise RuntimeError("commander server startup failed") from exc

    async def _await_started(self, server: Any) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.startup_timeout_s
        task = self._task
        assert task is not None
        while not getattr(server, "started", False):
            if task.done():
                await task  # re-raise whatever ended the server
                raise RuntimeError("commander server stopped before it finished starting")
            if loop.time() >= deadline:
                raise TimeoutError(
                    f"commander server did not start within {self.startup_timeout_s:g}s"
                )
            await asyncio.sleep(0.01)


__all__ = [
    "CommanderServer",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
]
