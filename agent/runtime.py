"""Production composition and lifecycle for the live agent runtime."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.bus import EventBus
from agent.config import Config
from agent.planning import Pipeliner, TaskAllocator
from agent.resync import StartupResync
from agent.sim import EntitySim
from agent.supervisor import TaskFactory, TaskSupervisor
from agent.telemetry.metrics import Metrics
from agent.telemetry.recorder import JsonlRecorder
from agent.transport.action_tracker import ActionTracker
from agent.transport.parsing import TOPIC_PARSED_MESSAGE, WireParser
from agent.transport.ws import GameSocket
from agent.world import Ingestor, ThreatMap, TrackStore, WorldModel, WorldPersistence

log = logging.getLogger("agent.runtime")


@dataclass(frozen=True, slots=True)
class PlanningContext:
    """Completed planning primitives exposed to an installed strategy."""

    allocator: TaskAllocator[Any]
    pipeliner: Pipeliner
    world: WorldModel
    tracks: TrackStore
    entity_sim: EntitySim
    threat_map: ThreatMap


class PlanningStrategy(Protocol):
    """Boundary implemented by future strategist/controller tickets."""

    async def start(self, context: PlanningContext) -> None: ...

    async def stop(self) -> None: ...


class PlanningBoundary:
    """Own planning lifecycle without enabling unfinished behavior by default."""

    def __init__(
        self,
        context: PlanningContext,
        *,
        enabled: bool = False,
        strategy: PlanningStrategy | None = None,
    ) -> None:
        self.context = context
        self.enabled = enabled
        self.strategy = strategy
        self._task: asyncio.Task[None] | None = None
        self._strategy_started = False

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    async def start(
        self,
        *,
        task_starter: Callable[[str, TaskFactory], asyncio.Task[None]] | None = None,
    ) -> None:
        if not self.enabled or self._task is not None:
            return
        if self.strategy is not None:
            self._strategy_started = True
            await self.strategy.start(self.context)
        if task_starter is None:
            self._task = asyncio.create_task(
                self.context.pipeliner.run(), name="planning-pipeliner"
            )
        else:
            self._task = task_starter("planning-pipeliner", self.context.pipeliner.run)
        # Let ``run`` initialise its stop event before a caller can immediately
        # request shutdown.
        await asyncio.sleep(0)

    async def stop(self) -> None:
        error: BaseException | None = None
        task = self._task
        if task is not None:
            try:
                await self.context.pipeliner.stop()
                await task
            except BaseException as exc:
                error = exc
            finally:
                self._task = None
        if self.strategy is not None and self._strategy_started:
            try:
                await self.strategy.stop()
            except BaseException as exc:
                error = error or exc
            finally:
                self._strategy_started = False
        if error is not None:
            raise error


@dataclass(slots=True)
class AgentRuntime:
    """All live subsystems plus their ordered startup and shutdown lifecycle."""

    config: Config
    rest: Any
    bus: EventBus
    socket: Any
    tracker: Any
    wire_parser: WireParser
    world: WorldModel
    tracks: TrackStore
    threat_map: ThreatMap
    ingestor: Ingestor
    entity_sim: EntitySim
    metrics: Metrics
    recorder: JsonlRecorder
    persistence: WorldPersistence
    pipeliner: Pipeliner
    allocator: TaskAllocator[Any]
    planning: PlanningBoundary
    resync: StartupResync
    supervisor: TaskSupervisor
    _unsubscribers: list[Callable[[], None]] = field(default_factory=list)
    _tasks: list[asyncio.Task[Any]] = field(default_factory=list)
    _started: bool = False
    _closed: bool = False
    restored_snapshot: bool = False

    def attach_consumers(self) -> None:
        """Connect parsed wire events to every stateful live consumer."""

        if self._unsubscribers:
            return
        self._unsubscribers.extend(
            (
                self.bus.subscribe(TOPIC_PARSED_MESSAGE, self._track_message),
                self.bus.subscribe(TOPIC_PARSED_MESSAGE, self.ingestor.on_message),
                self.bus.subscribe(TOPIC_PARSED_MESSAGE, self._simulate_message),
            )
        )

    @property
    def background_tasks(self) -> tuple[asyncio.Task[Any], ...]:
        planning_task = self.planning.task
        tasks = tuple(self._tasks)
        return tasks if planning_task is None else (*tasks, planning_task)

    async def start(self) -> None:
        """Restore and reconcile state before admitting live autonomy."""

        if self._closed:
            raise RuntimeError("runtime has already been closed")
        if self._started:
            return
        self._started = True
        try:
            self.restored_snapshot = await self.persistence.restore()
            if self.restored_snapshot:
                log.info(
                    "Restored world snapshot for %s at cycle %d",
                    self.config.persistence_match_id,
                    self.world.cycle,
                )
            await self.resync.run()
            self._tasks = [
                self.supervisor.start("world-persistence", self.persistence.run),
                self.supervisor.start("action-tracker", self.tracker.run),
                self.supervisor.start("game-socket", self.socket.run),
            ]
            await self.planning.start(task_starter=self.supervisor.start)
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop producers first, flush durable state, then detach consumers."""

        if self._closed:
            return
        errors: list[BaseException] = []

        async def attempt(operation: Callable[[], Awaitable[Any]]) -> None:
            try:
                await operation()
            except BaseException as exc:
                errors.append(exc)
                log.exception("Runtime shutdown operation failed")

        def close(operation: Callable[[], Any]) -> None:
            try:
                operation()
            except BaseException as exc:
                errors.append(exc)
                log.exception("Runtime close operation failed")

        # Mark shutdown first so normal service exits cannot be mistaken for
        # crashes and restarted while the rest of the runtime is stopping.
        self.supervisor.begin_shutdown()
        await attempt(self.planning.stop)
        await attempt(self.socket.stop)
        await attempt(self.tracker.stop)
        if self._started:
            await attempt(self.persistence.stop)
        await attempt(self.supervisor.stop)

        for unsubscribe in self._unsubscribers:
            close(unsubscribe)
        self._unsubscribers.clear()
        close(self.wire_parser.close)
        close(self.tracker.close)
        close(self.entity_sim.close)
        close(self.metrics.close)
        await attempt(self.recorder.close)
        await attempt(self.rest.aclose)

        self._tasks.clear()
        self._started = False
        self._closed = True
        if errors:
            raise RuntimeError("one or more runtime shutdown operations failed") from errors[0]

    async def _track_message(self, _topic: str, message: Any) -> None:
        await self.tracker.on_message(message)

    async def _simulate_message(self, _topic: str, message: Any) -> None:
        self.entity_sim.apply_message(message)


def compose_runtime(
    config: Config,
    rest: Any,
    *,
    gap_fill: Callable[[int], Awaitable[list[dict[str, Any]]]] | None = None,
    tracker_factory: Callable[..., Any] = ActionTracker,
    socket_factory: Callable[..., Any] = GameSocket,
    strategy: PlanningStrategy | None = None,
) -> AgentRuntime:
    """Construct every completed subsystem from one runtime configuration."""

    bus = EventBus()
    recorder = JsonlRecorder(config.telemetry_path, bus)
    metrics = Metrics(bus)
    wire_parser = WireParser(bus)
    tracker = tracker_factory(rest, bus, message_topic=None)
    world = WorldModel(bus)
    tracks = TrackStore(bus)
    ingestor = Ingestor(world, tracks)
    entity_sim = EntitySim()
    threat_map = ThreatMap(world, tracks)
    persistence = WorldPersistence(
        config.world_database,
        match_id=config.persistence_match_id,
        world=world,
        interval_s=config.snapshot_interval_s,
    )
    pipeliner = Pipeliner(rest, tracker, target_depth=config.queue_depth_target)
    allocator: TaskAllocator[Any] = TaskAllocator()
    context = PlanningContext(allocator, pipeliner, world, tracks, entity_sim, threat_map)
    planning = PlanningBoundary(
        context,
        enabled=config.planning_enabled or strategy is not None,
        strategy=strategy,
    )
    socket = socket_factory(
        config.server_url,
        token_provider=lambda: rest.token,
        bus=bus,
        gap_fill=gap_fill,
    )
    socket_resync = getattr(socket, "resync", None)
    resync = StartupResync(
        rest,
        gap_fill=(lambda: socket_resync(strict=True)) if callable(socket_resync) else None,
    )
    supervisor = TaskSupervisor(
        initial_backoff_s=config.restart_initial_backoff_s,
        max_backoff_s=config.restart_max_backoff_s,
    )
    runtime = AgentRuntime(
        config=config,
        rest=rest,
        bus=bus,
        socket=socket,
        tracker=tracker,
        wire_parser=wire_parser,
        world=world,
        tracks=tracks,
        threat_map=threat_map,
        ingestor=ingestor,
        entity_sim=entity_sim,
        metrics=metrics,
        recorder=recorder,
        persistence=persistence,
        pipeliner=pipeliner,
        allocator=allocator,
        planning=planning,
        resync=resync,
        supervisor=supervisor,
    )
    runtime.attach_consumers()
    return runtime


__all__ = [
    "AgentRuntime",
    "PlanningBoundary",
    "PlanningContext",
    "PlanningStrategy",
    "compose_runtime",
]
