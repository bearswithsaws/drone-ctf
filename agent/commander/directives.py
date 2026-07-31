"""Lifecycle management for commander directives.

Directives are intentionally kept outside the planner.  The store resolves the
latest command for each scope, expires it at ``issued_at + ttl_seconds``, and
publishes immutable effective-state snapshots.  Strategists can therefore
consume one coherent stance/order/override view without owning wall-clock
timers themselves.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

from agent.bus import EventBus
from agent.commander.contract import (
    Directive,
    EntityOverrideDirective,
    SquadOrderDirective,
    StanceDirective,
)


TOPIC_DIRECTIVES_CHANGED = "commander.directives.changed"

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """Normalise contract timestamps; treat a rare naive value as UTC."""

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _expires_at(directive: Directive) -> datetime:
    return _as_utc(directive.issued_at) + timedelta(seconds=directive.ttl_seconds)


@dataclass(frozen=True, slots=True)
class DirectiveState:
    """One immutable snapshot of the directives effective right now."""

    revision: int
    stance: StanceDirective | None
    squad_orders: Mapping[str, SquadOrderDirective]
    entity_overrides: Mapping[str, EntityOverrideDirective]

    def __post_init__(self) -> None:
        object.__setattr__(self, "squad_orders", MappingProxyType(dict(self.squad_orders)))
        object.__setattr__(
            self, "entity_overrides", MappingProxyType(dict(self.entity_overrides))
        )

    @property
    def pinned_tasks(self) -> Mapping[str, str]:
        """Return the allocator overlay implied by entity override directives."""

        return MappingProxyType(
            {
                entity_id: directive.task_id
                for entity_id, directive in self.entity_overrides.items()
            }
        )


class DirectiveStore:
    """Resolve directive scopes and publish changes, including TTL expiry."""

    def __init__(self, bus: EventBus, *, clock: Clock = _utc_now) -> None:
        self.bus = bus
        self._clock = clock
        self._stance: StanceDirective | None = None
        self._squad_orders: dict[str, SquadOrderDirective] = {}
        self._entity_overrides: dict[str, EntityOverrideDirective] = {}
        self._revision = 0
        self._wake = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def state(self) -> DirectiveState:
        """Return current effective state without racing the expiry publisher."""

        now = _as_utc(self._clock())
        stance = (
            self._stance
            if self._stance is not None and _expires_at(self._stance) > now
            else None
        )
        squad_orders = {
            squad_id: directive
            for squad_id, directive in self._squad_orders.items()
            if _expires_at(directive) > now
        }
        entity_overrides = {
            entity_id: directive
            for entity_id, directive in self._entity_overrides.items()
            if _expires_at(directive) > now
        }
        return DirectiveState(
            revision=self._revision,
            stance=stance,
            squad_orders=squad_orders,
            entity_overrides=entity_overrides,
        )

    async def apply(self, directive: Directive) -> DirectiveState:
        """Make ``directive`` effective for its scope and notify consumers."""

        if self._closed:
            raise RuntimeError("directive store is closed")

        self._expire_due()
        if isinstance(directive, StanceDirective):
            self._stance = directive
        elif isinstance(directive, SquadOrderDirective):
            self._squad_orders[directive.squad_id] = directive
        elif isinstance(directive, EntityOverrideDirective):
            self._entity_overrides[directive.entity_id] = directive
        else:  # pragma: no cover - the contract union prevents this at the API edge
            raise TypeError(f"unsupported directive type: {type(directive).__name__}")

        # Already-expired commands are acknowledged but never become effective.
        self._expire_due()
        self._revision += 1
        state = self._snapshot()
        await self.bus.publish(TOPIC_DIRECTIVES_CHANGED, state)
        self._ensure_worker()
        self._wake.set()
        return state

    async def refresh(self) -> DirectiveState:
        """Expire due directives immediately (also useful with an injected clock)."""

        changed = self._expire_due()
        if changed:
            self._revision += 1
            await self.bus.publish(TOPIC_DIRECTIVES_CHANGED, self._snapshot())
        self._wake.set()
        return self._snapshot()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._wake.set()
        worker = self._worker
        if worker is not None:
            try:
                await worker
            finally:
                self._worker = None

    def _snapshot(self) -> DirectiveState:
        return DirectiveState(
            revision=self._revision,
            stance=self._stance,
            squad_orders=self._squad_orders,
            entity_overrides=self._entity_overrides,
        )

    def _expire_due(self) -> bool:
        now = _as_utc(self._clock())
        changed = False
        if self._stance is not None and _expires_at(self._stance) <= now:
            self._stance = None
            changed = True
        for squad_id, directive in tuple(self._squad_orders.items()):
            if _expires_at(directive) <= now:
                del self._squad_orders[squad_id]
                changed = True
        for entity_id, directive in tuple(self._entity_overrides.items()):
            if _expires_at(directive) <= now:
                del self._entity_overrides[entity_id]
                changed = True
        return changed

    def _next_expiry(self) -> datetime | None:
        directives: list[Directive] = [
            *self._squad_orders.values(),
            *self._entity_overrides.values(),
        ]
        if self._stance is not None:
            directives.append(self._stance)
        return min((_expires_at(directive) for directive in directives), default=None)

    def _ensure_worker(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(
                self._expiry_worker(), name="commander-directive-expiry"
            )

    async def _expiry_worker(self) -> None:
        while not self._closed:
            self._wake.clear()
            next_expiry = self._next_expiry()
            if next_expiry is None:
                await self._wake.wait()
                continue
            delay = max(0.0, (next_expiry - _as_utc(self._clock())).total_seconds())
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=delay)
            except asyncio.TimeoutError:
                await self.refresh()


__all__ = [
    "DirectiveState",
    "DirectiveStore",
    "TOPIC_DIRECTIVES_CHANGED",
]
