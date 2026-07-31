"""Restart crashed asyncio services with bounded exponential backoff."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

log = logging.getLogger("agent.supervisor")

TaskFactory = Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class SupervisedTaskStatus:
    """Observable state for one supervised service."""

    name: str
    starts: int
    restarts: int
    last_failure: str | None


class TaskSupervisor:
    """Own long-running tasks and restart unexpected exits.

    A task factory is called again after either an exception or an unexpected
    clean return. Backoff resets after the task has remained healthy for
    ``stable_after_s``. Calling :meth:`begin_shutdown` before stopping the
    underlying services makes their normal return terminal instead.
    """

    def __init__(
        self,
        *,
        initial_backoff_s: float = 0.25,
        max_backoff_s: float = 5.0,
        stable_after_s: float = 30.0,
        shutdown_timeout_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if initial_backoff_s <= 0:
            raise ValueError("initial_backoff_s must be positive")
        if max_backoff_s < initial_backoff_s:
            raise ValueError("max_backoff_s must be at least initial_backoff_s")
        if stable_after_s <= 0:
            raise ValueError("stable_after_s must be positive")
        if shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be positive")
        self.initial_backoff_s = initial_backoff_s
        self.max_backoff_s = max_backoff_s
        self.stable_after_s = stable_after_s
        self.shutdown_timeout_s = shutdown_timeout_s
        self._clock = clock
        self._stopping = asyncio.Event()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._starts: dict[str, int] = {}
        self._restarts: dict[str, int] = {}
        self._last_failures: dict[str, str] = {}

    @property
    def tasks(self) -> tuple[asyncio.Task[None], ...]:
        return tuple(self._tasks.values())

    @property
    def stopping(self) -> bool:
        return self._stopping.is_set()

    def status(self, name: str) -> SupervisedTaskStatus | None:
        if name not in self._tasks:
            return None
        return SupervisedTaskStatus(
            name=name,
            starts=self._starts.get(name, 0),
            restarts=self._restarts.get(name, 0),
            last_failure=self._last_failures.get(name),
        )

    def start(self, name: str, factory: TaskFactory) -> asyncio.Task[None]:
        """Start supervising ``factory`` and return the supervisor task."""

        if self.stopping:
            raise RuntimeError("supervisor is stopping")
        existing = self._tasks.get(name)
        if existing is not None and not existing.done():
            raise ValueError(f"task {name!r} is already supervised")
        task = asyncio.create_task(self._supervise(name, factory), name=f"supervisor:{name}")
        self._tasks[name] = task
        return task

    def begin_shutdown(self) -> None:
        """Prevent restarts and wake services currently waiting in backoff."""

        self._stopping.set()

    async def stop(self) -> None:
        """Join stopped services, cancelling only those that miss the deadline."""

        self.begin_shutdown()
        tasks = self.tasks
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=self.shutdown_timeout_s)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            if not task.cancelled():
                task.exception()

    async def _supervise(self, name: str, factory: TaskFactory) -> None:
        failures = 0
        while not self.stopping:
            started_at = self._clock()
            self._starts[name] = self._starts.get(name, 0) + 1
            failure: Exception | None = None
            try:
                await factory()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failure = exc

            if self.stopping:
                return

            ran_for = max(0.0, self._clock() - started_at)
            if ran_for >= self.stable_after_s:
                failures = 0
            failures += 1
            delay = min(
                self.initial_backoff_s * (2 ** (failures - 1)),
                self.max_backoff_s,
            )
            self._restarts[name] = self._restarts.get(name, 0) + 1
            if failure is None:
                self._last_failures[name] = "unexpected clean exit"
                log.error("Supervised task %s exited; restarting in %.2fs", name, delay)
            else:
                self._last_failures[name] = f"{type(failure).__name__}: {failure}"
                log.error(
                    "Supervised task %s crashed; restarting in %.2fs",
                    name,
                    delay,
                    exc_info=(type(failure), failure, failure.__traceback__),
                )

            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass


__all__ = ["SupervisedTaskStatus", "TaskFactory", "TaskSupervisor"]
