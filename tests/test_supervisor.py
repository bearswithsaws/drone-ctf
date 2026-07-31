"""Failure-injection tests for asyncio task supervision."""

from __future__ import annotations

import asyncio

from agent.supervisor import TaskSupervisor


async def test_crashed_task_restarts_with_bounded_backoff() -> None:
    supervisor = TaskSupervisor(initial_backoff_s=0.005, max_backoff_s=0.01)
    healthy = asyncio.Event()
    release = asyncio.Event()
    attempts = 0

    async def flaky() -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError(f"failure {attempts}")
        healthy.set()
        await release.wait()

    task = supervisor.start("flaky", flaky)
    await asyncio.wait_for(healthy.wait(), timeout=0.2)

    status = supervisor.status("flaky")
    assert status is not None
    assert status.starts == 3
    assert status.restarts == 2
    assert status.last_failure == "RuntimeError: failure 2"

    supervisor.begin_shutdown()
    release.set()
    await supervisor.stop()
    assert task.done()


async def test_unexpected_clean_exit_is_restarted() -> None:
    supervisor = TaskSupervisor(initial_backoff_s=0.001, max_backoff_s=0.001)
    restarted = asyncio.Event()
    calls = 0

    async def exits() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            restarted.set()

    supervisor.start("short-lived", exits)
    await asyncio.wait_for(restarted.wait(), timeout=0.2)
    supervisor.begin_shutdown()
    await supervisor.stop()

    status = supervisor.status("short-lived")
    assert status is not None
    assert status.starts >= 2
    assert status.restarts >= 1


async def test_shutdown_interrupts_backoff_without_another_start() -> None:
    supervisor = TaskSupervisor(initial_backoff_s=1, max_backoff_s=1)
    crashed = asyncio.Event()
    calls = 0

    async def fails() -> None:
        nonlocal calls
        calls += 1
        crashed.set()
        raise RuntimeError("boom")

    task = supervisor.start("failing", fails)
    await asyncio.wait_for(crashed.wait(), timeout=0.2)
    await asyncio.sleep(0)
    supervisor.begin_shutdown()
    await supervisor.stop()

    assert calls == 1
    assert task.done()
