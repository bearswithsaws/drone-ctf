"""Local accounting for values omitted from action-completed messages.

The game only exposes battery, consumable, and building-resource totals in
explicit status actions.  Between status reports, :class:`EntitySim` combines
the action's queued details with its completion message and applies the pure
tables in :mod:`agent.rules.costs`.

Status completions are authoritative checkpoints.  Their values replace local
estimates (and are not charged a second time), which keeps replay drift bounded
while still giving controllers useful values between checkpoints.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from agent.bus import EventBus
from agent.rules.costs import (
    AUTO_CANNON_RELOAD_COST,
    BASELINE_TIER_COST,
    BUILDING_COSTS,
    DRONE_COST,
    EQUIPMENT_COSTS,
    LUCAS_COST,
    MISSILE_RELOAD_COST,
    REPAIR_COST,
    UNLOCK_COST,
    UPGRADE_TIER_COST,
    estimate_battery_cost,
)
from agent.transport.ws import TOPIC_MESSAGE


@dataclass
class DroneSimState:
    """Locally visible resource state for one drone."""

    drone_id: str
    current_battery: int
    max_battery: int
    total_weight: float = 0
    consumables: dict[str, int] = field(default_factory=dict)
    loaded_ammo_type: str | None = None


@dataclass
class BuildingSimState:
    """Locally visible resource state for one building."""

    building_id: str
    building_type: str | None = None
    stored_resources: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationDelta:
    """Accounting changes caused by one completed action."""

    action_id: str | None
    entity_id: str | None
    subject: str
    battery_spent: int = 0
    consumables_spent: dict[str, int] = field(default_factory=dict)
    resources_spent: dict[str, int] = field(default_factory=dict)
    status_checkpoint: bool = False
    duplicate: bool = False


@dataclass(frozen=True)
class _QueuedAction:
    subject: str
    details: dict[str, Any]


_CONSUMABLE_COUNT_FIELDS: dict[str, tuple[str, ...]] = {
    "wire_guided_missile": ("missiles_remaining", "remaining_missiles"),
    "missile_launcher": ("missiles_remaining", "remaining_missiles"),
    "land_mine": ("mines_remaining", "remaining_mines"),
    "mine_layer": ("mines_remaining", "remaining_mines"),
    "decoy_beacon": ("decoys_remaining", "remaining_decoys"),
    "smoke_launcher": ("charges_remaining", "remaining_charges"),
    "howitzer_shells": ("shells_remaining", "remaining_shells"),
    "howitzer_charges": ("charges_remaining", "remaining_charges"),
    "ammunition": ("rounds_remaining", "remaining_rounds", "current_ammo"),
    "armour_piercing_ammunition": (
        "rounds_remaining",
        "remaining_rounds",
        "current_ammo",
    ),
    "high_explosive_ammunition": (
        "rounds_remaining",
        "remaining_rounds",
        "current_ammo",
    ),
    "canister_round_ammunition": (
        "rounds_remaining",
        "remaining_rounds",
        "current_ammo",
    ),
}

_GENERIC_COUNT_FIELDS = (
    "remaining_uses",
    "uses_remaining",
    "remaining",
    "quantity",
    "count",
    "current_count",
)

_AMMO_TYPE_ALIASES = {
    "armour_piercing": "armour_piercing_ammunition",
    "armor_piercing": "armour_piercing_ammunition",
    "ap": "armour_piercing_ammunition",
    "high_explosive": "high_explosive_ammunition",
    "he": "high_explosive_ammunition",
    "canister_round": "canister_round_ammunition",
    "canister": "canister_round_ammunition",
}


class EntitySim:
    """Track stripped entity fields from status checkpoints and action events.

    The simulator accepts raw wire dictionaries or Pydantic wire models.  When
    attached to an :class:`~agent.bus.EventBus`, it consumes the same ``message``
    topic as the action tracker and telemetry recorder.
    """

    def __init__(
        self,
        bus: EventBus | None = None,
        *,
        research_tiers: Mapping[str, int] | None = None,
        action_cache_size: int = 10_000,
    ) -> None:
        if action_cache_size < 1:
            raise ValueError("action_cache_size must be positive")
        self._drones: dict[str, DroneSimState] = {}
        self._buildings: dict[str, BuildingSimState] = {}
        self._queued: OrderedDict[str, _QueuedAction] = OrderedDict()
        self._completed_action_ids: OrderedDict[str, None] = OrderedDict()
        self._action_cache_size = action_cache_size
        self._charged_research: set[tuple[str, int]] = set()
        self._research_tiers = {
            str(key): max(0, int(value)) for key, value in (research_tiers or {}).items()
        }
        self._unsubscribe = bus.subscribe(TOPIC_MESSAGE, self._on_bus_message) if bus else None

    @property
    def drones(self) -> dict[str, DroneSimState]:
        """Return a defensive snapshot of all locally tracked drones."""

        return deepcopy(self._drones)

    @property
    def buildings(self) -> dict[str, BuildingSimState]:
        """Return a defensive snapshot of all locally tracked buildings."""

        return deepcopy(self._buildings)

    @property
    def research_tiers(self) -> dict[str, int]:
        return dict(self._research_tiers)

    def get_drone(self, drone_id: str) -> DroneSimState | None:
        state = self._drones.get(drone_id)
        return deepcopy(state) if state is not None else None

    def get_building(self, building_id: str) -> BuildingSimState | None:
        state = self._buildings.get(building_id)
        return deepcopy(state) if state is not None else None

    def seed_drone(
        self,
        drone_id: str,
        *,
        current_battery: int,
        max_battery: int | None = None,
        total_weight: float = 0,
        consumables: Mapping[str, int] | None = None,
        loaded_ammo_type: str | None = None,
    ) -> DroneSimState:
        """Install an explicit starting checkpoint (useful for replay/tests)."""

        maximum = max(0, int(max_battery if max_battery is not None else current_battery))
        state = DroneSimState(
            drone_id=drone_id,
            current_battery=min(maximum, max(0, int(current_battery))),
            max_battery=maximum,
            total_weight=max(0.0, float(total_weight)),
            consumables={
                str(key): max(0, int(value)) for key, value in (consumables or {}).items()
            },
            loaded_ammo_type=_canonical_ammo_type(loaded_ammo_type),
        )
        self._drones[drone_id] = state
        return deepcopy(state)

    def seed_building(
        self,
        building_id: str,
        *,
        building_type: str | None = None,
        stored_resources: Mapping[str, int] | None = None,
    ) -> BuildingSimState:
        """Install an explicit building-resource checkpoint."""

        state = BuildingSimState(
            building_id=building_id,
            building_type=building_type,
            stored_resources={
                str(key): max(0, int(value)) for key, value in (stored_resources or {}).items()
            },
        )
        self._buildings[building_id] = state
        return deepcopy(state)

    def set_research_tiers(self, tiers: Mapping[str, int]) -> None:
        self._research_tiers = {str(key): max(0, int(value)) for key, value in tiers.items()}

    def remember_action(
        self,
        action_id: str,
        *,
        subject: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Supply action context when a replay omitted its queued message."""

        self._remember_queued(action_id, _QueuedAction(subject, dict(details or {})))

    def apply_message(self, message: Any) -> SimulationDelta | None:
        """Apply one raw/typed wire message and return a completion delta."""

        payload = _as_mapping(message)
        message_type = payload.get("message_type")
        action_id = _string_or_none(payload.get("action_id"))

        if message_type in {"action_queued", "building_action_queued"}:
            if action_id is not None:
                self._remember_queued(
                    action_id,
                    _QueuedAction(
                        str(payload.get("subject", "")),
                        _mapping_copy(payload.get("details")),
                    ),
                )
            return None

        if message_type not in {"action_completed", "building_action_completed"}:
            return None

        entity_id = _string_or_none(
            payload.get("drone_id")
            if message_type == "action_completed"
            else payload.get("building_id")
        )
        subject = str(payload.get("subject", ""))

        if action_id is not None and action_id in self._completed_action_ids:
            return SimulationDelta(
                action_id=action_id,
                entity_id=entity_id,
                subject=subject,
                duplicate=True,
            )

        queued = self._queued.pop(action_id, None) if action_id is not None else None
        details = dict(queued.details) if queued is not None else {}
        details.update(_mapping_copy(payload.get("details")))
        if not subject and queued is not None:
            subject = queued.subject.replace("queued", "completed")

        if action_id is not None:
            self._completed_action_ids[action_id] = None
            if len(self._completed_action_ids) > self._action_cache_size:
                self._completed_action_ids.popitem(last=False)

        if message_type == "action_completed":
            return self._apply_drone_completion(action_id, entity_id, subject, details)
        return self._apply_building_completion(action_id, entity_id, subject, details)

    def replay(self, records: Iterable[Any]) -> list[SimulationDelta]:
        """Replay wire messages or telemetry envelopes in input order."""

        deltas: list[SimulationDelta] = []
        for record in records:
            payload = _as_mapping(record)
            if "topic" in payload:
                if payload.get("topic") != TOPIC_MESSAGE:
                    continue
                payload = _as_mapping(payload.get("payload"))
            delta = self.apply_message(payload)
            if delta is not None:
                deltas.append(delta)
        return deltas

    def close(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    async def _on_bus_message(self, _topic: str, message: Any) -> None:
        self.apply_message(message)

    def _remember_queued(self, action_id: str, action: _QueuedAction) -> None:
        self._queued[action_id] = action
        self._queued.move_to_end(action_id)
        if len(self._queued) > self._action_cache_size:
            self._queued.popitem(last=False)

    def _apply_drone_completion(
        self,
        action_id: str | None,
        drone_id: str | None,
        subject: str,
        details: Mapping[str, Any],
    ) -> SimulationDelta:
        status = details.get("status")
        if _is_status_subject(subject) and isinstance(status, Mapping):
            state = self._checkpoint_drone(status, fallback_id=drone_id)
            return SimulationDelta(
                action_id=action_id,
                entity_id=state.drone_id if state is not None else drone_id,
                subject=subject,
                status_checkpoint=state is not None,
            )

        state = self._drones.get(drone_id) if drone_id is not None else None
        battery_spent = estimate_battery_cost(
            _battery_subject(subject),
            details,
            research_tiers=self._research_tiers,
            total_weight=state.total_weight if state is not None else 0,
        )
        if state is not None:
            state.current_battery = max(0, state.current_battery - battery_spent)

        spent = self._apply_consumable_effect(state, subject, details)
        self._apply_battery_transfer(state, subject, details)
        return SimulationDelta(
            action_id=action_id,
            entity_id=drone_id,
            subject=subject,
            battery_spent=battery_spent,
            consumables_spent=spent,
        )

    def _apply_building_completion(
        self,
        action_id: str | None,
        building_id: str | None,
        subject: str,
        details: Mapping[str, Any],
    ) -> SimulationDelta:
        status = details.get("status")
        if _is_status_subject(subject) and isinstance(status, Mapping):
            state = self._checkpoint_building(status, fallback_id=building_id)
            return SimulationDelta(
                action_id=action_id,
                entity_id=state.building_id if state is not None else building_id,
                subject=subject,
                status_checkpoint=state is not None,
            )

        if "status report completed" in _normalise_subject(subject):
            summary = details.get("summary")
            resources = (
                summary.get("command_center_resources")
                if isinstance(summary, Mapping)
                else None
            )
            if building_id is not None and isinstance(resources, Mapping):
                existing = self._buildings.get(building_id)
                building_type = existing.building_type if existing else "command_center"
                self.seed_building(
                    building_id,
                    building_type=building_type or "command_center",
                    stored_resources=resources,
                )
            return SimulationDelta(
                action_id=action_id,
                entity_id=building_id,
                subject=subject,
                status_checkpoint=isinstance(resources, Mapping),
            )

        actor = self._buildings.get(building_id) if building_id is not None else None
        cost = self._building_resource_cost(subject, details, actor)
        source = self._resource_source(actor)
        spent = _deduct_resources(source, cost) if source is not None else {}
        return SimulationDelta(
            action_id=action_id,
            entity_id=building_id,
            subject=subject,
            resources_spent=spent,
        )

    def _checkpoint_drone(
        self, status: Mapping[str, Any], *, fallback_id: str | None
    ) -> DroneSimState | None:
        drone_id = _string_or_none(status.get("drone_id")) or fallback_id
        if drone_id is None:
            return None

        prior = self._drones.get(drone_id)
        current = _int_value(status.get("current_battery"), prior.current_battery if prior else 0)
        maximum = _int_value(status.get("max_battery"), prior.max_battery if prior else current)
        equipment = status.get("equipment")
        equipment_items = equipment if isinstance(equipment, list) else []
        total_weight = max(0.0, _float_value(status.get("chassis_weight"), 0.0))
        consumables: dict[str, int] = {}
        loaded_ammo = prior.loaded_ammo_type if prior is not None else None

        for raw_item in equipment_items:
            item = _as_mapping(raw_item)
            equipment_type = _string_or_none(item.get("equipment_type"))
            if equipment_type is None:
                continue
            total_weight += max(0.0, _float_value(item.get("weight"), 0.0))
            if equipment_type in _CONSUMABLE_COUNT_FIELDS:
                count = _extract_consumable_count(equipment_type, item)
                if count is not None:
                    consumables[equipment_type] = consumables.get(equipment_type, 0) + count
            if equipment_type == "auto_cannon":
                candidate = (
                    item.get("loaded_ammo_type")
                    or item.get("loaded_ammunition_type")
                    or item.get("ammo_type")
                )
                if candidate is not None:
                    loaded_ammo = _canonical_ammo_type(str(candidate))

        state = DroneSimState(
            drone_id=drone_id,
            current_battery=min(max(0, maximum), max(0, current)),
            max_battery=max(0, maximum),
            total_weight=total_weight,
            consumables=consumables,
            loaded_ammo_type=loaded_ammo,
        )
        self._drones[drone_id] = state
        return state

    def _checkpoint_building(
        self, status: Mapping[str, Any], *, fallback_id: str | None
    ) -> BuildingSimState | None:
        building_id = _string_or_none(status.get("building_id")) or fallback_id
        if building_id is None:
            return None
        resources = status.get("stored_resources")
        state = BuildingSimState(
            building_id=building_id,
            building_type=_string_or_none(status.get("building_type")),
            stored_resources=_nonnegative_int_mapping(resources),
        )
        self._buildings[building_id] = state
        return state

    def _apply_consumable_effect(
        self,
        state: DroneSimState | None,
        subject: str,
        details: Mapping[str, Any],
    ) -> dict[str, int]:
        if state is None:
            return {}
        normalised = _normalise_subject(subject)

        if "autocannon load completed" in normalised or "auto cannon load completed" in normalised:
            ammo = _canonical_ammo_type(_string_or_none(details.get("ammo_type")))
            if ammo is not None:
                state.loaded_ammo_type = ammo
            return {}
        if (
            "autocannon unload completed" in normalised
            or "auto cannon unload completed" in normalised
        ):
            state.loaded_ammo_type = None
            return {}

        equipment_type: str | None = None
        amount = 1
        if (
            "fire cannon completed" in normalised
            or "autocannon fire completed" in normalised
            or "auto cannon fire completed" in normalised
        ):
            equipment_type = state.loaded_ammo_type or _first_present_ammo(state.consumables)
        elif "fire missile completed" in normalised or "missile fire completed" in normalised:
            equipment_type = _first_present(
                state.consumables,
                "wire_guided_missile",
                "missile_launcher",
            )
        elif "land mine deploy completed" in normalised or "mine deploy completed" in normalised:
            equipment_type = _first_present(state.consumables, "land_mine", "mine_layer")
        elif "decoy" in normalised and "deploy completed" in normalised:
            equipment_type = "decoy_beacon"
        elif "smoke" in normalised and "deploy completed" in normalised:
            equipment_type = "smoke_launcher"
        elif "howitzer load shell completed" in normalised:
            equipment_type = "howitzer_shells"
        elif "howitzer load charges completed" in normalised:
            equipment_type = "howitzer_charges"
            amount = max(0, _int_value(details.get("charges"), 1))
        elif "ammo transfer completed" in normalised:
            return self._apply_ammo_transfer(state, details)

        if equipment_type is None:
            return {}
        actual = _decrement_consumable(state, equipment_type, amount)
        return {equipment_type: actual} if actual else {}

    def _apply_ammo_transfer(
        self, source: DroneSimState, details: Mapping[str, Any]
    ) -> dict[str, int]:
        target_id = _string_or_none(details.get("target_drone_id"))
        target = self._drones.get(target_id) if target_id is not None else None
        equipment_type = _string_or_none(details.get("equipment_type"))
        if equipment_type is None:
            equipment_type = _string_or_none(details.get("ammo_type"))
        equipment_type = _canonical_ammo_type(equipment_type) or equipment_type
        if target is None or equipment_type not in source.consumables:
            return {}
        requested = _int_value(details.get("amount"), source.consumables[equipment_type])
        moved = _decrement_consumable(source, equipment_type, max(0, requested))
        target.consumables[equipment_type] = target.consumables.get(equipment_type, 0) + moved
        return {equipment_type: moved} if moved else {}

    def _apply_battery_transfer(
        self,
        source: DroneSimState | None,
        subject: str,
        details: Mapping[str, Any],
    ) -> None:
        if source is None or "charge completed" not in _normalise_subject(subject):
            return
        target_id = _string_or_none(details.get("target_drone_id"))
        target = self._drones.get(target_id) if target_id is not None else None
        if target is None:
            return
        amount = details.get("battery_transferred", details.get("charge_amount"))
        if amount is None:
            return
        target.current_battery = min(
            target.max_battery,
            target.current_battery + max(0, _int_value(amount, 0)),
        )

    def _resource_source(self, actor: BuildingSimState | None) -> BuildingSimState | None:
        for building in self._buildings.values():
            if building.building_type == "command_center":
                return building
        return actor

    def _building_resource_cost(
        self,
        subject: str,
        details: Mapping[str, Any],
        actor: BuildingSimState | None,
    ) -> dict[str, int]:
        explicit = details.get("resource_cost") or details.get("resources_spent")
        if isinstance(explicit, Mapping):
            return _nonnegative_int_mapping(explicit)

        normalised = _normalise_subject(subject)
        building_type = _string_or_none(details.get("building_type"))
        if "build" in normalised and building_type in BUILDING_COSTS:
            return dict(BUILDING_COSTS[building_type])

        if "manufacture" in normalised or "production" in normalised:
            equipment_type = _string_or_none(details.get("equipment_type"))
            if equipment_type in EQUIPMENT_COSTS:
                return dict(EQUIPMENT_COSTS[equipment_type])
            if actor is not None and actor.building_type == "lucas_launcher":
                return dict(LUCAS_COST)
            if actor is not None and actor.building_type == "drone_production_plant":
                return dict(DRONE_COST)

        if "reload" in normalised and actor is not None:
            if actor.building_type == "missile_turret":
                return dict(MISSILE_RELOAD_COST)
            if actor.building_type == "auto_cannon_turret":
                return dict(AUTO_CANNON_RELOAD_COST)

        if "repair" in normalised and (
            actor is None or actor.building_type not in {"drone_repair_station"}
        ):
            return dict(REPAIR_COST)

        if "research" in normalised:
            equipment_type = _string_or_none(details.get("equipment_type"))
            if equipment_type is None:
                return {}
            current = self._research_tiers.get(equipment_type, 0)
            reported_level = details.get("new_level", details.get("target_level"))
            new_level = max(current + 1, _int_value(reported_level, current + 1))
            research_key = (equipment_type, new_level)
            if research_key in self._charged_research:
                return {}
            self._charged_research.add(research_key)
            if "new_level" in details or details.get("research_completed") is True:
                self._research_tiers[equipment_type] = new_level
            if equipment_type.startswith("unlock_"):
                return dict(UNLOCK_COST)
            enriched = (
                BASELINE_TIER_COST.get(new_level, 0)
                if equipment_type == "baseline_drone"
                else UPGRADE_TIER_COST.get(new_level, 0)
            )
            return {"enriched_unobtanium": enriched} if enriched else {}

        return {}


def _as_mapping(value: Any) -> dict[str, Any]:
    raw_payload = getattr(value, "raw_payload", None)
    if isinstance(raw_payload, Mapping) and raw_payload:
        return dict(raw_payload)
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python")
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return {}


def _mapping_copy(value: Any) -> dict[str, Any]:
    return _as_mapping(value)


def _normalise_subject(subject: str) -> str:
    return " ".join(subject.lower().replace("_", " ").replace("-", " ").split())


def _battery_subject(subject: str) -> str:
    return _normalise_subject(subject).replace("auto cannon", "autocannon")


def _is_status_subject(subject: str) -> bool:
    normalised = _normalise_subject(subject)
    return "status completed" in normalised and "status report" not in normalised


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _nonnegative_int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): max(0, _int_value(item, 0)) for key, item in value.items()}


def _extract_consumable_count(
    equipment_type: str, item: Mapping[str, Any]
) -> int | None:
    fields = _CONSUMABLE_COUNT_FIELDS.get(equipment_type, ()) + _GENERIC_COUNT_FIELDS
    for field_name in fields:
        if field_name in item:
            return max(0, _int_value(item[field_name], 0))
    current_load = item.get("current_load")
    if isinstance(current_load, (int, float)) and not isinstance(current_load, bool):
        return max(0, int(current_load))
    if "capacity" in item:
        return max(0, _int_value(item["capacity"], 0))
    return None


def _canonical_ammo_type(value: str | None) -> str | None:
    if value is None:
        return None
    normalised = value.lower().replace("-", "_").replace(" ", "_")
    return _AMMO_TYPE_ALIASES.get(normalised, normalised)


def _first_present(consumables: Mapping[str, int], *keys: str) -> str:
    return next((key for key in keys if key in consumables), keys[0])


def _first_present_ammo(consumables: Mapping[str, int]) -> str | None:
    for key in (
        "armour_piercing_ammunition",
        "high_explosive_ammunition",
        "canister_round_ammunition",
        "ammunition",
    ):
        if key in consumables:
            return key
    return None


def _decrement_consumable(state: DroneSimState, equipment_type: str, amount: int) -> int:
    before = max(0, state.consumables.get(equipment_type, 0))
    actual = min(before, max(0, amount))
    state.consumables[equipment_type] = before - actual
    return actual


def _deduct_resources(
    state: BuildingSimState, cost: Mapping[str, int]
) -> dict[str, int]:
    spent: dict[str, int] = {}
    for resource, requested in cost.items():
        before = max(0, state.stored_resources.get(resource, 0))
        actual = min(before, max(0, int(requested)))
        state.stored_resources[resource] = before - actual
        if actual:
            spent[resource] = actual
    return spent


# Compatibility alias for callers that prefer the longer name.
EntitySimulator = EntitySim
