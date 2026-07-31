"""In-process telemetry counters and gauges.

The collector has no dependency on a metrics backend.  It consumes typed bus
events and exposes immutable snapshots that the future ops API can serialize
or export to Prometheus without changing producers.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import pairwise
from math import inf, isfinite
from typing import Any

from agent.bus import EventBus
from agent.transport.action_tracker import (
    TOPIC_ACTION_LOSS,
    TOPIC_ACTION_SUBMITTED,
    LossKind,
)

TOPIC_QUEUE_DEPTH = "queue.depth"
TOPIC_KILL_DEATH = "combat.kill_death"
TOPIC_BATTERY_MARGIN = "battery.margin"

DEFAULT_DISTANCE_BUCKETS = (5.0, 10.0, 20.0, 40.0)


@dataclass(frozen=True)
class QueueDepthEvent:
    entity_kind: str
    entity_id: str
    depth: int


@dataclass(frozen=True)
class KillDeathEvent:
    killer_id: str
    victim_id: str
    timestamp: float
    action_id: str | None = None


@dataclass(frozen=True)
class BatteryMarginEvent:
    entity_id: str
    current_battery: float
    required_battery: float = 0.0
    reserve_battery: float = 0.0
    max_battery: float | None = None


@dataclass(frozen=True)
class DistanceBucket:
    lower: float
    upper: float | None

    @property
    def label(self) -> str:
        lower = f"{self.lower:g}"
        return f"{lower}+" if self.upper is None else f"{lower}-{self.upper:g}"

    def contains(self, distance: float) -> bool:
        return distance >= self.lower and (self.upper is None or distance < self.upper)


@dataclass(frozen=True)
class LossRateSnapshot:
    bucket: DistanceBucket
    submissions: int
    losses: int
    losses_by_kind: Mapping[str, int]

    @property
    def loss_rate(self) -> float:
        return self.losses / self.submissions if self.submissions else 0.0


@dataclass
class _LossCounter:
    submissions: int = 0
    losses: int = 0
    losses_by_kind: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class CombatLedgerEntry:
    kills: int = 0
    deaths: int = 0


@dataclass(frozen=True)
class BatteryMargin:
    current_battery: float
    required_battery: float
    reserve_battery: float
    max_battery: float | None

    @property
    def margin(self) -> float:
        return self.current_battery - self.required_battery - self.reserve_battery

    @property
    def margin_fraction(self) -> float | None:
        if self.max_battery is None or self.max_battery <= 0:
            return None
        return self.margin / self.max_battery


class Metrics:
    """Collect communication, queue, combat, and battery telemetry."""

    def __init__(
        self,
        bus: EventBus | None = None,
        *,
        distance_buckets: tuple[float, ...] = DEFAULT_DISTANCE_BUCKETS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        edges = tuple(float(edge) for edge in distance_buckets)
        if any(not isfinite(edge) or edge <= 0 for edge in edges) or any(
            left >= right for left, right in pairwise(edges)
        ):
            raise ValueError("distance bucket edges must be positive and strictly increasing")

        bounds = (0.0, *edges, inf)
        self._buckets = tuple(
            DistanceBucket(lower, None if upper == inf else upper)
            for lower, upper in pairwise(bounds)
        )
        self._loss = {bucket: _LossCounter() for bucket in self._buckets}
        self._queue_depths: dict[str, dict[str, int]] = {}
        self._ledger: dict[str, dict[str, int]] = {}
        self._kill_events: list[KillDeathEvent] = []
        self._battery: dict[str, BatteryMargin] = {}
        self._clock = clock
        self._unsubscribers: list[Callable[[], None]] = []
        if bus is not None:
            self.attach(bus)

    def attach(self, bus: EventBus) -> None:
        self._unsubscribers.extend(
            (
                bus.subscribe(TOPIC_ACTION_SUBMITTED, self._on_action_submitted),
                bus.subscribe(TOPIC_ACTION_LOSS, self._on_action_loss),
                bus.subscribe(TOPIC_QUEUE_DEPTH, self._on_queue_depth),
                bus.subscribe(TOPIC_KILL_DEATH, self._on_kill_death),
                bus.subscribe(TOPIC_BATTERY_MARGIN, self._on_battery_margin),
            )
        )

    def close(self) -> None:
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()

    def _bucket_for(self, distance: float) -> DistanceBucket:
        value = float(distance)
        if not isfinite(value):
            raise ValueError("distance must be finite")
        value = max(0.0, value)
        return next(bucket for bucket in self._buckets if bucket.contains(value))

    def record_submission(self, distance: float) -> None:
        self._loss[self._bucket_for(distance)].submissions += 1

    def record_loss(self, distance: float, kind: LossKind | str) -> None:
        counter = self._loss[self._bucket_for(distance)]
        counter.losses += 1
        value = _string_value(kind)
        counter.losses_by_kind[value] += 1

    async def _on_action_submitted(self, _topic: str, event: Any) -> None:
        distance = _field(event, "distance")
        if distance is not None:
            self.record_submission(float(distance))

    async def _on_action_loss(self, _topic: str, event: Any) -> None:
        distance = _field(event, "distance")
        kind = _field(event, "kind")
        if distance is not None and kind is not None:
            self.record_loss(float(distance), kind)

    @property
    def loss_rate_by_distance(self) -> Mapping[str, LossRateSnapshot]:
        return {
            bucket.label: LossRateSnapshot(
                bucket=bucket,
                submissions=counter.submissions,
                losses=counter.losses,
                losses_by_kind=dict(counter.losses_by_kind),
            )
            for bucket, counter in self._loss.items()
        }

    # Explicit observations allow the queue manager, combat ingest, and entity
    # simulator to report authoritative values as those components land.
    def observe_queue_depth(self, entity_kind: str, entity_id: str, depth: int) -> None:
        value = int(depth)
        if value < 0:
            raise ValueError("queue depth cannot be negative")
        kind = _string_value(entity_kind)
        self._queue_depths.setdefault(kind, {})[str(entity_id)] = value

    async def _on_queue_depth(self, _topic: str, event: Any) -> None:
        self.observe_queue_depth(
            _required_field(event, "entity_kind"),
            str(_required_field(event, "entity_id")),
            int(_required_field(event, "depth")),
        )

    @property
    def queue_depths(self) -> Mapping[str, Mapping[str, int]]:
        return {kind: dict(depths) for kind, depths in self._queue_depths.items()}

    def record_kill(
        self,
        killer_id: str,
        victim_id: str,
        *,
        action_id: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        event = KillDeathEvent(
            killer_id=str(killer_id),
            victim_id=str(victim_id),
            timestamp=self._clock() if timestamp is None else float(timestamp),
            action_id=action_id,
        )
        self._ledger.setdefault(event.killer_id, {"kills": 0, "deaths": 0})["kills"] += 1
        self._ledger.setdefault(event.victim_id, {"kills": 0, "deaths": 0})["deaths"] += 1
        self._kill_events.append(event)

    async def _on_kill_death(self, _topic: str, event: Any) -> None:
        self.record_kill(
            str(_required_field(event, "killer_id")),
            str(_required_field(event, "victim_id")),
            action_id=_field(event, "action_id"),
            timestamp=_field(event, "timestamp"),
        )

    @property
    def kill_death_ledger(self) -> Mapping[str, CombatLedgerEntry]:
        return {
            entity_id: CombatLedgerEntry(**counts) for entity_id, counts in self._ledger.items()
        }

    @property
    def kill_events(self) -> tuple[KillDeathEvent, ...]:
        return tuple(self._kill_events)

    def observe_battery_margin(
        self,
        entity_id: str,
        current_battery: float,
        *,
        required_battery: float = 0.0,
        reserve_battery: float = 0.0,
        max_battery: float | None = None,
    ) -> None:
        self._battery[str(entity_id)] = BatteryMargin(
            current_battery=float(current_battery),
            required_battery=float(required_battery),
            reserve_battery=float(reserve_battery),
            max_battery=None if max_battery is None else float(max_battery),
        )

    async def _on_battery_margin(self, _topic: str, event: Any) -> None:
        self.observe_battery_margin(
            str(_required_field(event, "entity_id")),
            float(_required_field(event, "current_battery")),
            required_battery=float(_field(event, "required_battery", 0.0)),
            reserve_battery=float(_field(event, "reserve_battery", 0.0)),
            max_battery=_field(event, "max_battery"),
        )

    @property
    def battery_margins(self) -> Mapping[str, BatteryMargin]:
        return dict(self._battery)

    def snapshot(self) -> dict[str, Any]:
        """Return a fully JSON-serializable point-in-time snapshot."""

        return {
            "comms_loss_by_distance": {
                label: {
                    "lower": item.bucket.lower,
                    "upper": item.bucket.upper,
                    "submissions": item.submissions,
                    "losses": item.losses,
                    "loss_rate": item.loss_rate,
                    "losses_by_kind": dict(item.losses_by_kind),
                }
                for label, item in self.loss_rate_by_distance.items()
            },
            "queue_depths": {kind: dict(depths) for kind, depths in self.queue_depths.items()},
            "kill_death_ledger": {
                entity_id: asdict(entry) for entity_id, entry in self.kill_death_ledger.items()
            },
            "battery_margins": {
                entity_id: {
                    **asdict(margin),
                    "margin": margin.margin,
                    "margin_fraction": margin.margin_fraction,
                }
                for entity_id, margin in self.battery_margins.items()
            },
        }


def _field(event: Any, name: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(name, default)
    return getattr(event, name, default)


def _required_field(event: Any, name: str) -> Any:
    value = _field(event, name)
    if value is None:
        raise ValueError(f"telemetry event is missing {name!r}")
    return value


def _string_value(value: Any) -> str:
    return str(value.value) if isinstance(value, Enum) else str(value)


TelemetryMetrics = Metrics
