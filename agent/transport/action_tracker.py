"""Idempotent action submission and three-signal loss reconciliation.

The game reports an action through three independent channels: the POST
response, an ``*_action_queued`` message, and an ``*_action_completed`` (or
``action_failed``) message.  Radio loss can remove either message, and an
uncertain POST may not have reached the server at all.  :class:`ActionTracker`
keeps a controller intent stable across those failures and consults the
server-side queue before it ever repeats a command.

Controllers must supply a precondition callback.  It is evaluated immediately
before a lost command is resubmitted; a stale intent is abandoned instead of
causing a second movement or other unsafe action.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping, Protocol

from agent.bus import EventBus
from agent.transport.ws import TOPIC_MESSAGE

log = logging.getLogger("agent.transport.action_tracker")

TOPIC_ACTION_LIFECYCLE = "action.lifecycle"
TOPIC_ACTION_LOSS = "action.loss"

_QUEUED_MESSAGE_TYPES = {"action_queued", "building_action_queued"}
_COMPLETED_MESSAGE_TYPES = {"action_completed", "building_action_completed"}
_FAILED_MESSAGE_TYPES = {"action_failed", "building_action_failed"}


class EntityKind(str, Enum):
    DRONE = "drone"
    BUILDING = "building"

    @property
    def queue_path(self) -> str:
        return "/queue/drones" if self is EntityKind.DRONE else "/queue/buildings"

    @property
    def endpoint_segment(self) -> str:
        return "drones" if self is EntityKind.DRONE else "buildings"


class IntentState(str, Enum):
    SUBMITTED = "submitted"
    QUEUED = "queued"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"

    @property
    def terminal(self) -> bool:
        return self in {IntentState.COMPLETED, IntentState.FAILED, IntentState.ABANDONED}


class LossKind(str, Enum):
    COMMAND = "command_lost"
    QUEUED_REPORT = "queued_report_lost"
    COMPLETION_REPORT = "completion_report_lost"


Precondition = Callable[[], bool | Awaitable[bool]]
AckTimeout = Callable[[float], float]
Clock = Callable[[], float]


class RestResult(Protocol):
    data: dict[str, Any]

    @property
    def ok(self) -> bool: ...


class RestClient(Protocol):
    async def get(self, path: str, params: dict[str, Any] | None = None) -> RestResult: ...

    async def post(self, path: str, json: dict[str, Any] | None = None) -> RestResult: ...


def default_ack_timeout(distance: float) -> float:
    """Default distance-sensitive acknowledgement window, in seconds.

    The coefficients are deliberately conservative policy defaults.  E4's
    comms calibration can inject its learned timeout function instead.
    """

    return 1.0 + max(0.0, float(distance)) * 0.05


@dataclass
class Intent:
    local_id: str
    entity_kind: EntityKind
    entity_id: str
    action: str
    endpoint: str
    payload: dict[str, Any]
    distance: float
    precondition: Precondition = field(repr=False, compare=False)
    state: IntentState = IntentState.SUBMITTED
    action_id: str | None = None
    server_action_ids: list[str] = field(default_factory=list)
    attempts: int = 0
    submitted_at: float | None = None
    queued_at: float | None = None
    terminal_at: float | None = None
    deadline_at: float | None = None
    last_message: dict[str, Any] | None = None
    saw_queued: bool = False
    inferred_terminal: bool = False

    @property
    def terminal(self) -> bool:
        return self.state.terminal


@dataclass(frozen=True)
class ActionLifecycleEvent:
    local_id: str
    entity_kind: EntityKind
    entity_id: str
    action: str
    action_id: str | None
    previous_state: IntentState | None
    state: IntentState
    reason: str
    attempt: int
    timestamp: float
    inferred: bool = False


@dataclass(frozen=True)
class ActionLossEvent:
    local_id: str
    entity_kind: EntityKind
    entity_id: str
    action: str
    action_id: str | None
    kind: LossKind
    distance: float
    attempt: int
    timestamp: float


PendingEvent = tuple[str, ActionLifecycleEvent | ActionLossEvent]


class ActionTracker:
    """Registry and reconciliation loop for controller action intents.

    ``submit`` is idempotent by ``local_id``.  Reusing an id for different
    action data raises ``ValueError``; reusing it for the same action returns
    the existing intent without issuing another POST.
    """

    def __init__(
        self,
        rest: RestClient,
        bus: EventBus | None = None,
        *,
        ack_timeout: AckTimeout = default_ack_timeout,
        clock: Clock = time.monotonic,
        cycle_seconds: float = 0.25,
        reconcile_interval_s: float = 0.25,
        orphan_limit: int = 1000,
    ) -> None:
        self._rest = rest
        self._bus = bus
        self._ack_timeout = ack_timeout
        self._clock = clock
        self._cycle_seconds = cycle_seconds
        self._reconcile_interval = reconcile_interval_s
        self._orphan_limit = orphan_limit

        self._intents: dict[str, Intent] = {}
        self._by_action_id: dict[str, Intent] = {}
        self._orphans: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._unsubscribe = bus.subscribe(TOPIC_MESSAGE, self._on_bus_message) if bus else None

    @property
    def intents(self) -> Mapping[str, Intent]:
        """A shallow, read-only snapshot of the local-id registry."""

        return {local_id: replace(intent) for local_id, intent in self._intents.items()}

    def get(self, local_id: str) -> Intent | None:
        intent = self._intents.get(local_id)
        return replace(intent) if intent is not None else None

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
    ) -> Intent:
        """Register and send an action, returning its current intent snapshot."""

        kind = EntityKind(entity_kind)
        action = action.strip("/")
        resolved_endpoint = endpoint or f"/{kind.endpoint_segment}/{entity_id}/{action}"
        body = dict(payload or {})
        pending: list[PendingEvent] = []

        async with self._lock:
            existing = self._intents.get(local_id)
            if existing is not None:
                signature = (
                    existing.entity_kind,
                    existing.entity_id,
                    existing.action,
                    existing.endpoint,
                    existing.payload,
                )
                requested = (kind, entity_id, action, resolved_endpoint, body)
                if signature != requested:
                    raise ValueError(f"local_id {local_id!r} already belongs to another action")
                snapshot = replace(existing)
            else:
                intent = Intent(
                    local_id=local_id,
                    entity_kind=kind,
                    entity_id=entity_id,
                    action=action,
                    endpoint=resolved_endpoint,
                    payload=body,
                    distance=float(distance),
                    precondition=precondition,
                )
                self._intents[local_id] = intent
                await self._submit_attempt(intent, pending, reason="submitted")
                snapshot = replace(intent)

        await self._publish(pending)
        return snapshot

    async def on_message(self, message: Any) -> None:
        """Consume a raw dict or typed wire message from the socket transport."""

        payload = self._normalise_message(message)
        message_type = payload.get("message_type")
        action_id = payload.get("action_id")
        if not isinstance(action_id, str) or message_type not in (
            _QUEUED_MESSAGE_TYPES | _COMPLETED_MESSAGE_TYPES | _FAILED_MESSAGE_TYPES
        ):
            return

        pending: list[PendingEvent] = []
        async with self._lock:
            intent = self._by_action_id.get(action_id)
            if intent is None:
                messages = self._orphans.setdefault(action_id, [])
                messages.append(payload)
                self._orphans.move_to_end(action_id)
                while len(self._orphans) > self._orphan_limit:
                    self._orphans.popitem(last=False)
                return
            self._apply_message(intent, payload, pending)

        await self._publish(pending)

    async def reconcile(self) -> None:
        """Run one queue-diff pass for all intents whose deadline has elapsed."""

        pending: list[PendingEvent] = []
        async with self._lock:
            now = self._clock()
            due_by_kind: dict[EntityKind, list[Intent]] = {}
            for intent in self._intents.values():
                if (
                    intent.state in {IntentState.SUBMITTED, IntentState.QUEUED}
                    and intent.deadline_at is not None
                    and intent.deadline_at <= now
                ):
                    due_by_kind.setdefault(intent.entity_kind, []).append(intent)

            for kind, intents in due_by_kind.items():
                try:
                    result = await self._rest.get(kind.queue_path)
                except Exception as exc:
                    log.warning("Queue poll raised for %s actions: %s", kind.value, exc)
                    result = None

                if result is None or not result.ok:
                    for intent in intents:
                        intent.deadline_at = now + self._timeout(intent)
                        self._record_lifecycle(
                            pending, intent, intent.state, "queue_poll_failed"
                        )
                    continue

                present = self._extract_action_ids(result.data.get("actions", []))
                for intent in intents:
                    known_present = next(
                        (
                            action_id
                            for action_id in intent.server_action_ids
                            if action_id in present
                        ),
                        None,
                    )
                    if intent.state is IntentState.SUBMITTED:
                        if known_present is not None:
                            self._record_loss(
                                pending, intent, LossKind.QUEUED_REPORT, known_present
                            )
                            self._mark_queued(
                                intent,
                                pending,
                                action_id=known_present,
                                reason="queue_poll_found_action",
                                inferred=True,
                            )
                        else:
                            await self._handle_lost_command(intent, pending)
                    elif known_present is not None:
                        # It is still executing or waiting in the server queue.
                        intent.deadline_at = now + self._timeout(intent)
                    else:
                        self._record_loss(
                            pending, intent, LossKind.COMPLETION_REPORT, intent.action_id
                        )
                        self._mark_terminal(
                            intent,
                            pending,
                            IntentState.COMPLETED,
                            reason="queue_poll_action_disappeared",
                            inferred=True,
                        )

        await self._publish(pending)

    async def run(self) -> None:
        """Reconcile periodically until :meth:`stop` is called."""

        self._stop.clear()
        while not self._stop.is_set():
            await self.reconcile()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._reconcile_interval)
            except asyncio.TimeoutError:
                pass

    async def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    async def _on_bus_message(self, _topic: str, message: Any) -> None:
        await self.on_message(message)

    async def _submit_attempt(
        self, intent: Intent, pending: list[PendingEvent], *, reason: str
    ) -> None:
        previous = intent.state if intent.attempts else None
        intent.state = IntentState.SUBMITTED
        intent.inferred_terminal = False
        intent.attempts += 1
        intent.submitted_at = self._clock()
        intent.deadline_at = intent.submitted_at + self._timeout(intent)
        self._record_lifecycle(pending, intent, previous, reason)

        try:
            result = await self._rest.post(intent.endpoint, json=dict(intent.payload))
        except Exception as exc:
            log.warning("Action POST raised for %s: %s", intent.local_id, exc)
            self._record_lifecycle(pending, intent, intent.state, "submission_unacknowledged")
            return

        action_id = result.data.get("action_id") if result.ok else None
        if not isinstance(action_id, str) or not action_id:
            self._record_lifecycle(pending, intent, intent.state, "submission_unacknowledged")
            return

        self._bind_action_id(intent, action_id)
        self._record_lifecycle(pending, intent, intent.state, "server_action_id_assigned")
        for orphan in self._orphans.pop(action_id, []):
            self._apply_message(intent, orphan, pending)

    async def _handle_lost_command(
        self, intent: Intent, pending: list[PendingEvent]
    ) -> None:
        self._record_loss(pending, intent, LossKind.COMMAND, intent.action_id)
        try:
            holds = intent.precondition()
            if inspect.isawaitable(holds):
                holds = await holds
            precondition_holds = bool(holds)
        except Exception as exc:
            log.exception("Precondition raised for %s: %s", intent.local_id, exc)
            precondition_holds = False

        if not precondition_holds:
            self._mark_terminal(
                intent,
                pending,
                IntentState.ABANDONED,
                reason="precondition_changed",
                inferred=False,
            )
            return
        await self._submit_attempt(intent, pending, reason="command_lost_resubmitted")

    def _apply_message(
        self, intent: Intent, message: dict[str, Any], pending: list[PendingEvent]
    ) -> None:
        message_type = message.get("message_type")
        action_id = message.get("action_id")
        if not isinstance(action_id, str):
            return
        self._bind_action_id(intent, action_id)

        if message_type in _QUEUED_MESSAGE_TYPES:
            if intent.state.terminal:
                return
            self._mark_queued(
                intent,
                pending,
                action_id=action_id,
                reason="queued_message",
                inferred=False,
                message=message,
            )
            return

        terminal_state = (
            IntentState.COMPLETED
            if message_type in _COMPLETED_MESSAGE_TYPES
            else IntentState.FAILED
        )

        # Definitive terminal reports may correct a queue-based inference or a
        # prior abandonment caused by an apparently lost command.
        if (
            intent.state.terminal
            and not intent.inferred_terminal
            and intent.state is not IntentState.ABANDONED
        ):
            return
        if not intent.saw_queued:
            self._record_loss(pending, intent, LossKind.QUEUED_REPORT, action_id)
        self._mark_terminal(
            intent,
            pending,
            terminal_state,
            reason="terminal_message",
            inferred=False,
            message=message,
        )

    def _mark_queued(
        self,
        intent: Intent,
        pending: list[PendingEvent],
        *,
        action_id: str,
        reason: str,
        inferred: bool,
        message: dict[str, Any] | None = None,
    ) -> None:
        previous = intent.state
        intent.state = IntentState.QUEUED
        intent.action_id = action_id
        intent.saw_queued = True
        intent.queued_at = self._clock()
        intent.last_message = message
        cycles = self._cycles_from_message(message)
        intent.deadline_at = intent.queued_at + cycles * self._cycle_seconds + self._timeout(intent)
        self._record_lifecycle(pending, intent, previous, reason, inferred=inferred)

    def _mark_terminal(
        self,
        intent: Intent,
        pending: list[PendingEvent],
        state: IntentState,
        *,
        reason: str,
        inferred: bool,
        message: dict[str, Any] | None = None,
    ) -> None:
        previous = intent.state
        intent.state = state
        intent.inferred_terminal = inferred
        intent.terminal_at = self._clock()
        intent.deadline_at = None
        if message is not None:
            intent.last_message = message
            action_id = message.get("action_id")
            if isinstance(action_id, str):
                intent.action_id = action_id
        self._record_lifecycle(pending, intent, previous, reason, inferred=inferred)

    def _bind_action_id(self, intent: Intent, action_id: str) -> None:
        owner = self._by_action_id.get(action_id)
        if owner is not None and owner is not intent:
            raise ValueError(f"server action_id {action_id!r} is already bound")
        self._by_action_id[action_id] = intent
        if action_id not in intent.server_action_ids:
            intent.server_action_ids.append(action_id)
        intent.action_id = action_id

    def _timeout(self, intent: Intent) -> float:
        return max(0.0, float(self._ack_timeout(intent.distance)))

    @staticmethod
    def _cycles_from_message(message: dict[str, Any] | None) -> float:
        if message is None:
            return 0.0
        details = message.get("details")
        if not isinstance(details, dict):
            return 0.0
        cycles = details.get("cycles_required", details.get("cycles", 0))
        try:
            return max(0.0, float(cycles))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _normalise_message(message: Any) -> dict[str, Any]:
        if isinstance(message, dict):
            return dict(message)
        model_dump = getattr(message, "model_dump", None)
        if callable(model_dump):
            value = model_dump()
            if isinstance(value, dict):
                return value
        return {}

    @classmethod
    def _extract_action_ids(cls, actions: Any) -> set[str]:
        """Accept flat arrays and entity-keyed queue payload variants."""

        found: set[str] = set()
        if isinstance(actions, list):
            for item in actions:
                found.update(cls._extract_action_ids(item))
        elif isinstance(actions, dict):
            action_id = actions.get("action_id")
            if isinstance(action_id, str):
                found.add(action_id)
            else:
                for value in actions.values():
                    found.update(cls._extract_action_ids(value))
        elif isinstance(actions, str):
            # Some test/private server builds returned bare action-id arrays.
            found.add(actions)
        return found

    def _record_lifecycle(
        self,
        pending: list[PendingEvent],
        intent: Intent,
        previous: IntentState | None,
        reason: str,
        *,
        inferred: bool = False,
    ) -> None:
        pending.append(
            (
                TOPIC_ACTION_LIFECYCLE,
                ActionLifecycleEvent(
                    local_id=intent.local_id,
                    entity_kind=intent.entity_kind,
                    entity_id=intent.entity_id,
                    action=intent.action,
                    action_id=intent.action_id,
                    previous_state=previous,
                    state=intent.state,
                    reason=reason,
                    attempt=intent.attempts,
                    timestamp=self._clock(),
                    inferred=inferred,
                ),
            )
        )

    def _record_loss(
        self,
        pending: list[PendingEvent],
        intent: Intent,
        kind: LossKind,
        action_id: str | None,
    ) -> None:
        pending.append(
            (
                TOPIC_ACTION_LOSS,
                ActionLossEvent(
                    local_id=intent.local_id,
                    entity_kind=intent.entity_kind,
                    entity_id=intent.entity_id,
                    action=intent.action,
                    action_id=action_id,
                    kind=kind,
                    distance=intent.distance,
                    attempt=intent.attempts,
                    timestamp=self._clock(),
                ),
            )
        )

    async def _publish(self, pending: list[PendingEvent]) -> None:
        if self._bus is None:
            return
        for topic, event in pending:
            await self._bus.publish(topic, event)
