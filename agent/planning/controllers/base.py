"""Controllers for the base's economy buildings.

The game gives every building an independent FIFO action queue.  The
controllers in this module therefore expose one controller output per building
rather than trying to impose a single, cross-building queue:

* :class:`RefinerController` plans a bounded batch of one-unit process actions;
* :class:`ResearchController` is an inexhaustible action factory, because the
  command centre must receive the same research action every cycle; and
* :class:`ProductionController` expands a build order into the two production
  plant streams while preserving the order within each plant.

All outputs can be passed directly to :class:`agent.planning.pipeliner.Pipeliner`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from agent.planning.pipeliner import ControllerOutput, PlannedAction
from agent.planning.tasks import Research
from agent.rules.economy import REFINING_RULES, UNLOCK_RULES


Inventory: TypeAlias = Mapping[str, int]
InventorySource: TypeAlias = Inventory | Callable[[], Inventory]
ResearchTiers: TypeAlias = Mapping[str, int]
ResearchTierSource: TypeAlias = ResearchTiers | Callable[[], ResearchTiers]


def _non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} cannot be empty")


def _positive_int(value: int, field_name: str, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field_name} must be a {qualifier} integer")


def _efficiency(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.01 <= value <= 1.0
    ):
        raise ValueError("efficiency must be finite and between 0.01 and 1.0")
    return float(value)


def _read_source(source: InventorySource | ResearchTierSource) -> Mapping[str, int]:
    values = source() if callable(source) else source
    if not isinstance(values, Mapping):
        raise TypeError("state source must return a mapping")
    return values


@dataclass(frozen=True, slots=True)
class RefinerBatch:
    """A bounded group of ore-processing actions for one refiner."""

    refiner_id: str
    ore_type: str | None
    units: int
    actions: tuple[PlannedAction, ...]

    @property
    def empty(self) -> bool:
        return not self.actions


class RefinerController:
    """Select raw ore by priority and process at most one configured batch.

    The API processes exactly one ore unit per action.  A batch is consequently
    represented by several identical actions, capped by both ``batch_size`` and
    the latest known inventory.  Call :meth:`plan` again after the batch is
    exhausted (normally after a fresh building checkpoint).
    """

    def __init__(
        self,
        refiner_id: str,
        *,
        batch_size: int = 6,
        ore_priority: Sequence[str] = tuple(REFINING_RULES),
        efficiency: float = 1.0,
    ) -> None:
        _non_empty(refiner_id, "refiner_id")
        _positive_int(batch_size, "batch_size")
        priority = tuple(ore_priority)
        if not priority:
            raise ValueError("ore_priority cannot be empty")
        if any(not isinstance(ore, str) or not ore.strip() for ore in priority):
            raise ValueError("ore_priority entries must be non-empty strings")
        if len(set(priority)) != len(priority):
            raise ValueError("ore_priority cannot contain duplicates")

        self.refiner_id = refiner_id
        self.batch_size = batch_size
        self.ore_priority = priority
        self.efficiency = _efficiency(efficiency)

    def plan(self, inventory: InventorySource) -> RefinerBatch:
        """Plan the highest-priority non-empty ore batch from ``inventory``."""

        current = _read_source(inventory)
        selected: str | None = None
        available = 0
        for ore_type in self.ore_priority:
            amount = current.get(ore_type, 0)
            if isinstance(amount, bool) or not isinstance(amount, int):
                raise TypeError(f"inventory amount for {ore_type!r} must be an integer")
            if amount > 0:
                selected = ore_type
                available = amount
                break

        if selected is None:
            return RefinerBatch(self.refiner_id, None, 0, ())

        units = min(self.batch_size, available)

        def ore_remains(ore_type: str = selected) -> bool:
            # A callable source is live and protects ActionTracker retries after
            # a checkpoint reports that another consumer emptied the refiner.
            if not callable(inventory):
                return True
            amount = _read_source(inventory).get(ore_type, 0)
            return isinstance(amount, int) and not isinstance(amount, bool) and amount > 0

        actions = tuple(
            PlannedAction(
                "refiner/process",
                {"efficiency": self.efficiency, "ore_type": selected},
                precondition=ore_remains,
            )
            for _ in range(units)
        )
        return RefinerBatch(self.refiner_id, selected, units, actions)

    plan_batch = plan


@dataclass(frozen=True, slots=True)
class ResearchStep:
    """One research path and the tier the controller should reach."""

    research_key: str
    target_level: int

    def __post_init__(self) -> None:
        _non_empty(self.research_key, "research_key")
        _positive_int(self.target_level, "target_level")
        if self.research_key in UNLOCK_RULES and self.target_level != 1:
            raise ValueError("unlock research has exactly one tier")
        if self.research_key not in UNLOCK_RULES and self.target_level > 3:
            raise ValueError("equipment research target_level cannot exceed 3")


ResearchOrderEntry: TypeAlias = ResearchStep | Research | str


def _research_step(entry: ResearchOrderEntry) -> ResearchStep:
    if isinstance(entry, ResearchStep):
        return entry
    if isinstance(entry, Research):
        target = entry.target_level
        if target is None:
            target = 1 if entry.research_key in UNLOCK_RULES else 3
        return ResearchStep(entry.research_key, target)
    if isinstance(entry, str):
        return ResearchStep(entry, 1 if entry in UNLOCK_RULES else 3)
    raise TypeError("research order entries must be ResearchStep, Research, or str")


class ResearchController:
    """Continuously feed a command centre from an ordered research programme.

    ``next_action`` is intentionally a factory, not a finite iterator.  Once
    every configured target is met it keeps returning ``fallback_key`` (or the
    final configured key), so the pipeliner never marks the command-centre plan
    exhausted.  A live ``tiers`` callback lets the selected step advance as
    completion checkpoints arrive.
    """

    def __init__(
        self,
        command_center_id: str,
        research_order: Sequence[ResearchOrderEntry],
        tiers: ResearchTierSource,
        *,
        fallback_key: str | None = None,
        efficiency: float = 1.0,
    ) -> None:
        _non_empty(command_center_id, "command_center_id")
        order = tuple(_research_step(entry) for entry in research_order)
        if not order and fallback_key is None:
            raise ValueError("research_order cannot be empty without a fallback_key")
        keys = [step.research_key for step in order]
        if len(set(keys)) != len(keys):
            raise ValueError("research_order cannot contain duplicate research keys")
        if fallback_key is not None:
            _non_empty(fallback_key, "fallback_key")

        self.command_center_id = command_center_id
        self.research_order = order
        self._tiers = tiers
        self.fallback_key = fallback_key or order[-1].research_key
        self.efficiency = _efficiency(efficiency)

    def current_step(self) -> ResearchStep | None:
        """Return the first research target not yet met."""

        tiers = _read_source(self._tiers)
        for step in self.research_order:
            level = tiers.get(step.research_key, 0)
            if isinstance(level, bool) or not isinstance(level, int):
                raise TypeError(
                    f"research tier for {step.research_key!r} must be an integer"
                )
            if level < step.target_level:
                return step
        return None

    def next_action(self) -> PlannedAction:
        """Return the next per-cycle research action; this never returns ``None``."""

        step = self.current_step()
        research_key = step.research_key if step is not None else self.fallback_key
        assert research_key is not None

        def still_needed(step: ResearchStep | None = step) -> bool:
            if step is None:
                # The fallback exists specifically to keep the queue alive.
                return True
            level = _read_source(self._tiers).get(step.research_key, 0)
            return isinstance(level, int) and not isinstance(level, bool) and (
                level < step.target_level
            )

        return PlannedAction(
            "command_center/research",
            {"efficiency": self.efficiency, "equipment_type": research_key},
            precondition=still_needed,
        )

    __call__ = next_action

    @property
    def output(self) -> ControllerOutput:
        """The inexhaustible factory to register with a pipeliner."""

        return self.next_action


@dataclass(frozen=True, slots=True)
class BuildOrder:
    """Desired drone count and ordered equipment production.

    Repeating an equipment name requests multiple copies.  ``from_loadouts``
    is convenient when the strategist naturally describes each future drone by
    its equipment tuple.
    """

    drones: int = 0
    equipment: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _positive_int(self.drones, "drones", allow_zero=True)
        equipment = tuple(self.equipment)
        if any(not isinstance(item, str) or not item.strip() for item in equipment):
            raise ValueError("equipment entries must be non-empty strings")
        object.__setattr__(self, "equipment", equipment)

    @classmethod
    def from_loadouts(cls, loadouts: Sequence[Sequence[str]]) -> BuildOrder:
        """Build one drone per loadout and manufacture all listed equipment."""

        frozen = tuple(tuple(loadout) for loadout in loadouts)
        return cls(len(frozen), tuple(item for loadout in frozen for item in loadout))


@dataclass(frozen=True, slots=True)
class ProductionPlan:
    """Finite controller outputs for the two production plant queues."""

    drone_plant_id: str | None
    equipment_plant_id: str | None
    drone_actions: tuple[PlannedAction, ...]
    equipment_actions: tuple[PlannedAction, ...]

    def actions_for(self, building_id: str) -> tuple[PlannedAction, ...]:
        if building_id == self.drone_plant_id:
            return self.drone_actions
        if building_id == self.equipment_plant_id:
            return self.equipment_actions
        return ()

    @property
    def outputs(self) -> Mapping[str, tuple[PlannedAction, ...]]:
        """Return non-empty ``building_id -> output`` entries for registration."""

        values: dict[str, tuple[PlannedAction, ...]] = {}
        if self.drone_plant_id is not None and self.drone_actions:
            values[self.drone_plant_id] = self.drone_actions
        if self.equipment_plant_id is not None and self.equipment_actions:
            values[self.equipment_plant_id] = self.equipment_actions
        return MappingProxyType(values)


class ProductionController:
    """Expand one build order for drone and equipment production plants."""

    def __init__(
        self,
        drone_plant_id: str | None,
        equipment_plant_id: str | None,
        *,
        efficiency: float = 1.0,
    ) -> None:
        if drone_plant_id is not None:
            _non_empty(drone_plant_id, "drone_plant_id")
        if equipment_plant_id is not None:
            _non_empty(equipment_plant_id, "equipment_plant_id")
        if (
            drone_plant_id is not None
            and equipment_plant_id is not None
            and drone_plant_id == equipment_plant_id
        ):
            raise ValueError("drone and equipment plants must have different ids")
        self.drone_plant_id = drone_plant_id
        self.equipment_plant_id = equipment_plant_id
        self.efficiency = _efficiency(efficiency)

    def plan(self, build_order: BuildOrder) -> ProductionPlan:
        if not isinstance(build_order, BuildOrder):
            raise TypeError("build_order must be a BuildOrder")
        if build_order.drones and self.drone_plant_id is None:
            raise ValueError("build order requests drones but no drone plant is configured")
        if build_order.equipment and self.equipment_plant_id is None:
            raise ValueError("build order requests equipment but no equipment plant is configured")

        drone_actions = tuple(
            PlannedAction("production/manufacture", {"efficiency": self.efficiency})
            for _ in range(build_order.drones)
        )
        equipment_actions = tuple(
            PlannedAction(
                "production/manufacture",
                {"efficiency": self.efficiency, "equipment_type": equipment_type},
            )
            for equipment_type in build_order.equipment
        )
        return ProductionPlan(
            self.drone_plant_id,
            self.equipment_plant_id,
            drone_actions,
            equipment_actions,
        )


__all__ = [
    "BuildOrder",
    "ProductionController",
    "ProductionPlan",
    "RefinerBatch",
    "RefinerController",
    "ResearchController",
    "ResearchOrderEntry",
    "ResearchStep",
]
