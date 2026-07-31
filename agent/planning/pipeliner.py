"""Keep server-side entity action queues saturated from controller output.

Controllers produce :class:`PlannedAction` values; the pipeliner keeps a
small FIFO runway on the game server so a completed action can roll directly
into the next one.  Each submitted action still goes through
``ActionTracker`` for idempotency and loss reconciliation.

Replacing a plan is deliberately a two-phase operation: invalidate the old
generation and delete all of its server actions, then start consuming the new
controller output.  The generation guard is included in every tracker
precondition, preventing a lost command from being retried after its plan was
invalidated.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Awaitable, Protocol

from agent.transport.action_tracker import EntityKind, Precondition

log = logging.getLogger("agent.planning.pipeliner")

DEFAULT_MIN_DEPTH = 4
DEFAULT_TARGET_DEPTH = 6
DEFAULT_MAX_DEPTH = 8


async def _always() -> bool:
    return True


@dataclass(frozen=True, slots=True)
class PlannedAction:
    """One controller-produced action ready for server submission."""

    action: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    precondition: Precondition = field(default=_always, repr=False, compare=False)
    distance: float = 0.0
    endpoint: str | None = None

    def __post_init__(self) -> None:
        if not self.action.strip("/"):
            raise ValueError("action cannot be empty")
        if self.distance < 0:
            raise ValueError("distance cannot be negative")


ActionValue = PlannedAction | Mapping[str, Any]
ActionFactory = Callable[[], ActionValue | None | Awaitable[ActionValue | None]]
ControllerOutput = Iterable[ActionValue] | AsyncIterable[ActionValue] | ActionFactory


class RestResult(Protocol):
    data: dict[str, Any]

    @property
    def ok(self) -> bool: ...


class RestClient(Protocol):
    async def get(self, path: str, params: dict[str, Any] | None = None) -> RestResult: ...

    async def delete(self, path: str, json: dict[str, Any] | None = None) -> RestResult: ...


class TrackedIntent(Protocol):
    action_id: str | None


class Tracker(Protocol):
    async def submit(
        self,
        local_id: str,
        entity_kind: EntityKind | str,
        entity_id: str,
        action: str,
        *,
        precondition: Precondition,
        payload: Mapping[str, Any] | None = None,
        distance: float = 0.0,
        endpoint: str | None = None,
    ) -> TrackedIntent: ...


EntityKey = tuple[EntityKind, str]
_MISSING = object()


class PipelineError(RuntimeError):
    """Base class for server queue maintenance failures."""


class QueuePollError(PipelineError):
    """The authoritative server queue could not be read."""


class QueueFlushError(PipelineError):
    """A plan could not be safely rebuilt because its old queue was not cleared."""

    def __init__(self, entity_kind: EntityKind, entity_id: str, action_ids: Iterable[str]) -> None:
        self.entity_kind = entity_kind
        self.entity_id = entity_id
        self.action_ids = tuple(action_ids)
        joined = ", ".join(self.action_ids) or "unknown actions"
        super().__init__(f"could not flush {entity_kind.value} {entity_id}: {joined}")


@dataclass(frozen=True, slots=True)
class PipelineStatus:
    entity_kind: EntityKind
    entity_id: str
    generation: int
    queue_depth: int
    submitted: int
    exhausted: bool
    paused: bool


@dataclass(frozen=True, slots=True)
class InvalidationResult:
    entity_kind: EntityKind
    entity_id: str
    generation: int
    cancelled_action_ids: tuple[str, ...]
    rebuilt: int
    queue_depth: int


class _ActionSource:
    def __init__(self, output: ControllerOutput) -> None:
        self._factory: ActionFactory | None = output if callable(output) else None
        self._async_iterator: AsyncIterator[ActionValue] | None = None
        self._iterator: Iterator[ActionValue] | None = None
        if self._factory is None:
            if isinstance(output, AsyncIterable):
                self._async_iterator = aiter(output)
            else:
                if isinstance(output, (str, bytes, Mapping)):
                    raise TypeError(
                        "controller output must be an iterable of actions, not one value"
                    )
                self._iterator = iter(output)

    async def next(self) -> tuple[bool, PlannedAction | None]:
        """Return ``(exhausted, action)``; a factory may temporarily return None."""

        if self._factory is not None:
            value = self._factory()
            if inspect.isawaitable(value):
                value = await value
            return False, None if value is None else _normalise_action(value)
        if self._async_iterator is not None:
            try:
                return False, _normalise_action(await anext(self._async_iterator))
            except StopAsyncIteration:
                return True, None
        assert self._iterator is not None
        sentinel = object()
        value = next(self._iterator, sentinel)
        if value is sentinel:
            return True, None
        return False, _normalise_action(value)


@dataclass(slots=True)
class _Plan:
    kind: EntityKind
    entity_id: str
    source: _ActionSource
    generation: int
    sequence: int = 0
    queue_depth: int = 0
    exhausted: bool = False
    paused: bool = False


class Pipeliner:
    """Maintain bounded, non-empty server queues for registered entities.

    ``cycle`` polls each relevant queue once and fills every registered entity
    toward ``target_depth``.  ``replace_plan`` is the normal invalidation API:
    it flushes only the selected entity, installs the new controller output,
    and immediately rebuilds that entity's queue.
    """

    def __init__(
        self,
        rest: RestClient,
        tracker: Tracker,
        *,
        min_depth: int = DEFAULT_MIN_DEPTH,
        target_depth: int = DEFAULT_TARGET_DEPTH,
        max_depth: int = DEFAULT_MAX_DEPTH,
        interval_s: float = 0.25,
    ) -> None:
        if min_depth < 1:
            raise ValueError("min_depth must be positive")
        if not min_depth <= target_depth <= max_depth:
            raise ValueError("queue depths must satisfy min_depth <= target_depth <= max_depth")
        if not DEFAULT_MIN_DEPTH <= target_depth <= DEFAULT_MAX_DEPTH:
            raise ValueError("target_depth must be between 4 and 8")
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")

        self._rest = rest
        self._tracker = tracker
        self.min_depth = min_depth
        self.target_depth = target_depth
        self.max_depth = max_depth
        self._interval = interval_s
        self._plans: dict[EntityKey, _Plan] = {}
        self._generations: dict[EntityKey, int] = {}
        self._action_owners: dict[str, EntityKey] = {}
        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()

    def register(
        self,
        entity_kind: EntityKind | str,
        entity_id: str,
        output: ControllerOutput,
    ) -> None:
        """Register controller output for an entity with no existing plan."""

        kind = EntityKind(entity_kind)
        if not entity_id:
            raise ValueError("entity_id cannot be empty")
        key = (kind, entity_id)
        if key in self._plans:
            raise ValueError(f"{kind.value} {entity_id} already has a plan; use replace_plan")
        generation = self._generations.get(key, 0) + 1
        self._generations[key] = generation
        self._plans[key] = _Plan(kind, entity_id, _ActionSource(output), generation)

    set_plan = register

    def status(self, entity_kind: EntityKind | str, entity_id: str) -> PipelineStatus | None:
        """Return local plan state (queue depth is authoritative only after maintenance)."""

        kind = EntityKind(entity_kind)
        plan = self._plans.get((kind, entity_id))
        if plan is None:
            return None
        return PipelineStatus(
            kind,
            entity_id,
            plan.generation,
            plan.queue_depth,
            0,
            plan.exhausted,
            plan.paused,
        )

    async def cycle(self) -> list[PipelineStatus]:
        """Poll queues and replenish all registered entities once."""

        async with self._lock:
            statuses: list[PipelineStatus] = []
            for kind in EntityKind:
                plans = [plan for plan in self._plans.values() if plan.kind is kind]
                if not plans:
                    continue
                actions = await self._get_queue(kind)
                for plan in plans:
                    entity_actions = self._for_entity(actions, plan)
                    statuses.append(await self._fill(plan, len(entity_actions)))
            return statuses

    async def maintain(
        self, entity_kind: EntityKind | str, entity_id: str
    ) -> PipelineStatus:
        """Poll and replenish one entity, useful for event-driven wakeups."""

        kind = EntityKind(entity_kind)
        async with self._lock:
            plan = self._require_plan(kind, entity_id)
            actions = await self._get_queue(kind)
            return await self._fill(plan, len(self._for_entity(actions, plan)))

    async def invalidate(
        self,
        entity_kind: EntityKind | str,
        entity_id: str,
        replacement: ControllerOutput | object = _MISSING,
    ) -> InvalidationResult:
        """Flush an invalid plan and optionally install/rebuild its replacement.

        Without ``replacement`` the entity remains paused after its queue is
        cleared.  Passing replacement output (including an empty iterable)
        performs the full flush-and-rebuild operation.
        """

        kind = EntityKind(entity_kind)
        key = (kind, entity_id)
        async with self._lock:
            plan = self._require_plan(kind, entity_id)

            # Invalidate tracker retry preconditions before touching the server.
            generation = self._generations.get(key, plan.generation) + 1
            self._generations[key] = generation
            plan.generation = generation
            plan.paused = True

            actions = await self._get_queue(kind)
            doomed = self._for_entity(actions, plan)
            cancelled: list[str] = []
            failed: list[str] = []
            for action in doomed:
                action_id = _action_id(action)
                if action_id is None:
                    failed.append("<missing action_id>")
                    continue
                result = await self._rest.delete(f"{kind.queue_path}/{action_id}")
                status = getattr(result, "status", None)
                if result.ok or status == 404 or (
                    isinstance(status, int) and 200 <= status < 300
                ):
                    cancelled.append(action_id)
                    self._action_owners.pop(action_id, None)
                else:
                    failed.append(action_id)

            remaining = self._for_entity(await self._get_queue(kind), plan)
            failed.extend(
                action_id
                for action in remaining
                if (action_id := _action_id(action)) is not None and action_id not in failed
            )
            if any(_action_id(action) is None for action in remaining):
                failed.append("<missing action_id>")
            if failed:
                plan.queue_depth = len(remaining)
                raise QueueFlushError(kind, entity_id, failed)

            rebuilt = 0
            depth = 0
            plan.queue_depth = 0
            if replacement is not _MISSING:
                plan.source = _ActionSource(replacement)  # type: ignore[arg-type]
                plan.sequence = 0
                plan.exhausted = False
                plan.paused = False
                filled = await self._fill(plan, 0)
                rebuilt = filled.submitted
                depth = filled.queue_depth

            return InvalidationResult(
                kind,
                entity_id,
                plan.generation,
                tuple(cancelled),
                rebuilt,
                depth,
            )

    async def replace_plan(
        self,
        entity_kind: EntityKind | str,
        entity_id: str,
        output: ControllerOutput,
    ) -> InvalidationResult:
        """Flush the current entity queue and immediately build a new plan."""

        return await self.invalidate(entity_kind, entity_id, output)

    async def run(self) -> None:
        """Maintain all queues periodically until :meth:`stop` is called."""

        self._stop.clear()
        while not self._stop.is_set():
            try:
                await self.cycle()
            except PipelineError as exc:
                log.warning("Queue maintenance failed: %s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass

    async def stop(self) -> None:
        self._stop.set()

    async def _get_queue(self, kind: EntityKind) -> list[Mapping[str, Any]]:
        try:
            result = await self._rest.get(kind.queue_path)
        except Exception as exc:
            raise QueuePollError(f"could not poll {kind.value} queue: {exc}") from exc
        if not result.ok:
            raise QueuePollError(f"could not poll {kind.value} queue")
        actions = result.data.get("actions", [])
        if not isinstance(actions, list) or not all(
            isinstance(action, Mapping) for action in actions
        ):
            raise QueuePollError(f"malformed {kind.value} queue response")
        return actions

    def _for_entity(
        self, actions: Iterable[Mapping[str, Any]], plan: _Plan
    ) -> list[Mapping[str, Any]]:
        key = (plan.kind, plan.entity_id)
        matched: list[Mapping[str, Any]] = []
        for action in actions:
            action_id = _action_id(action)
            owner = self._action_owners.get(action_id) if action_id is not None else None
            entity_id = _entity_id(action, plan.kind)
            if entity_id == plan.entity_id or (entity_id is None and owner == key):
                matched.append(action)
                if action_id is not None:
                    self._action_owners[action_id] = key
        return matched

    async def _fill(self, plan: _Plan, queue_depth: int) -> PipelineStatus:
        submitted = 0
        if not plan.paused and not plan.exhausted and queue_depth < self.target_depth:
            desired = min(self.target_depth, self.max_depth)
            while queue_depth < desired:
                exhausted, action = await plan.source.next()
                if exhausted:
                    plan.exhausted = True
                    break
                if action is None:
                    break

                plan.sequence += 1
                generation = plan.generation
                key = (plan.kind, plan.entity_id)

                async def precondition(
                    action: PlannedAction = action,
                    generation: int = generation,
                    key: EntityKey = key,
                ) -> bool:
                    current = self._plans.get(key)
                    if current is None or current.generation != generation or current.paused:
                        return False
                    holds = action.precondition()
                    if inspect.isawaitable(holds):
                        holds = await holds
                    return bool(holds)

                local_id = (
                    f"pipeline:{plan.kind.value}:{plan.entity_id}:"
                    f"{generation}:{plan.sequence}"
                )
                intent = await self._tracker.submit(
                    local_id,
                    plan.kind,
                    plan.entity_id,
                    action.action,
                    precondition=precondition,
                    payload=dict(action.payload),
                    distance=action.distance,
                    endpoint=action.endpoint,
                )
                if intent.action_id:
                    self._action_owners[intent.action_id] = key
                # Even an unacknowledged POST may have reached the server. Count
                # it until the next authoritative poll to avoid an overfill burst.
                queue_depth += 1
                submitted += 1

        plan.queue_depth = queue_depth
        return PipelineStatus(
            plan.kind,
            plan.entity_id,
            plan.generation,
            queue_depth,
            submitted,
            plan.exhausted,
            plan.paused,
        )

    def _require_plan(self, kind: EntityKind, entity_id: str) -> _Plan:
        try:
            return self._plans[(kind, entity_id)]
        except KeyError as exc:
            raise KeyError(f"no plan registered for {kind.value} {entity_id}") from exc


def _normalise_action(value: ActionValue) -> PlannedAction:
    if isinstance(value, PlannedAction):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("controller actions must be PlannedAction values or mappings")
    action = value.get("action") or value.get("subject")
    if not isinstance(action, str):
        raise ValueError("controller action mapping must include an action string")
    payload = value.get("payload", value.get("details", {}))
    if not isinstance(payload, Mapping):
        raise ValueError("controller action payload must be a mapping")
    precondition = value.get("precondition", _always)
    if not callable(precondition):
        raise ValueError("controller action precondition must be callable")
    return PlannedAction(
        action=action,
        payload=dict(payload),
        precondition=precondition,
        distance=float(value.get("distance", 0.0)),
        endpoint=value.get("endpoint"),
    )


def _action_id(action: Mapping[str, Any]) -> str | None:
    value = action.get("action_id", action.get("id"))
    return value if isinstance(value, str) and value else None


def _entity_id(action: Mapping[str, Any], kind: EntityKind) -> str | None:
    keys = ("drone_id", "entity_id") if kind is EntityKind.DRONE else ("building_id", "entity_id")
    for source in (action, action.get("details"), action.get("parameters")):
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            value = source.get(key)
            if value is not None:
                return str(value)
    return None


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MIN_DEPTH",
    "DEFAULT_TARGET_DEPTH",
    "ControllerOutput",
    "InvalidationResult",
    "PipelineError",
    "PipelineStatus",
    "Pipeliner",
    "PlannedAction",
    "QueueFlushError",
    "QueuePollError",
]
