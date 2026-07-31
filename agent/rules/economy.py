"""Queryable economy and research rules.

This module collects the planning-oriented economy data that is spread across
the game guide and API documentation.  It deliberately keeps calculations
pure: callers can ask what something costs or how long it takes without
needing a world model or a connection to the game server.

Cycle counts are the base values at efficiency 1.0.  The server may take
longer when a building action is queued at a lower efficiency.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from agent.rules.costs import BUILDING_COSTS as _BASE_BUILDING_COSTS

ResourceCost = Mapping[str, int]


def _cost(**resources: int) -> ResourceCost:
    """Create an immutable resource-cost mapping."""

    return MappingProxyType(resources)


@dataclass(frozen=True)
class TimedCost:
    """A resource cost and its base completion time."""

    cost: ResourceCost
    cycles: int


@dataclass(frozen=True)
class BuildingRule:
    """Construction cost and base action timers for a buildable building."""

    cost: ResourceCost
    action_cycles: Mapping[str, int]


@dataclass(frozen=True)
class UnlockRule:
    """Single-tier research that unlocks one or more equipment products."""

    unlocks: tuple[str, ...]
    cost: ResourceCost
    cycles: int


@dataclass(frozen=True)
class RefiningRule:
    """Probabilistic conversion of one unit of raw ore."""

    output_resource: str
    success_probability: float
    cycles_per_unit: int = 1


# Building action names use the current API vocabulary.  The alias table also
# accepts the server's older/internal action names found in completion events.
_COMMON_BUILDING_ACTION_CYCLES = MappingProxyType({"status": 1, "repair": 20})

_BUILDING_ACTION_ALIASES = MappingProxyType(
    {
        "load_drone": "load",
        "unload_drone": "unload",
        "build_drone": "manufacture",
        "build_equipment": "manufacture",
        "build_lucas_drone": "manufacture",
        "install_equipment": "install",
        "remove_equipment": "remove",
        "repair_drone": "repair",
        "shield_generate": "generate",
        "transfer_shield": "transfer",
        "fire_missile": "fire",
        "fire_cannon": "fire",
        "building_pds_activate": "activate",
        "launch_lucas": "launch",
        "change_lucas_target": "redirect",
    }
)

_BUILDING_ACTION_CYCLES: dict[str, Mapping[str, int]] = {
    "solar_panel_array": MappingProxyType({"reposition": 1}),
    "battery_array": MappingProxyType({}),
    "drone_charging_station": MappingProxyType({"charge": 10}),
    "drone_repair_station": MappingProxyType({"repair": 5, "repair_equipment": 5}),
    "silo": MappingProxyType({"load": 5, "unload": 5}),
    "refiner": MappingProxyType({"process": 1, "load": 5, "unload": 5}),
    "drone_production_plant": MappingProxyType(
        {"manufacture": 500, "load": 5, "unload": 5}
    ),
    "drone_equipment_production_plant": MappingProxyType(
        {
            "manufacture": 25,
            "install": 25,
            "remove": 25,
            "load": 5,
            "unload": 5,
        }
    ),
    "shield_array": MappingProxyType({"generate": 1, "transfer": 1}),
    "laser_turret": MappingProxyType(
        {"fire": 1, "scan": 1, "identify": 1, "mark": 1, "load": 5, "unload": 5}
    ),
    "missile_turret": MappingProxyType(
        {
            "fire": 1,
            "reload": 40,
            "scan": 1,
            "identify": 1,
            "mark": 1,
            "load": 5,
            "unload": 5,
        }
    ),
    "auto_cannon_turret": MappingProxyType(
        {
            "fire": 1,
            "reload": 40,
            "scan": 1,
            "identify": 1,
            "mark": 1,
            "load": 5,
            "unload": 5,
        }
    ),
    "sensor_array": MappingProxyType({"scan": 1, "identify": 1}),
    "point_defense_system": MappingProxyType({"activate": 1}),
    "lucas_launcher": MappingProxyType(
        {"manufacture": 500, "launch": 25, "redirect": 1, "load": 5, "unload": 5}
    ),
}


BUILDING_RULES: Mapping[str, BuildingRule] = MappingProxyType(
    {
        building_type: BuildingRule(
            cost=MappingProxyType(dict(cost)),
            action_cycles=_BUILDING_ACTION_CYCLES[building_type],
        )
        for building_type, cost in _BASE_BUILDING_COSTS.items()
    }
)


_UNLOCK_COST = _cost(enriched_unobtanium=4)
_UNLOCK_CYCLES = 400

UNLOCK_RULES: Mapping[str, UnlockRule] = MappingProxyType(
    {
        "unlock_reactive_armour": UnlockRule(
            ("reactive_armour",), _UNLOCK_COST, _UNLOCK_CYCLES
        ),
        "unlock_hover_tech": UnlockRule(("hover_tech",), _UNLOCK_COST, _UNLOCK_CYCLES),
        "unlock_wire_guided_missile": UnlockRule(
            ("wire_guided_missile",), _UNLOCK_COST, _UNLOCK_CYCLES
        ),
        "unlock_auto_cannon": UnlockRule(
            (
                "auto_cannon",
                "armour_piercing_ammunition",
                "high_explosive_ammunition",
                "canister_round_ammunition",
            ),
            _UNLOCK_COST,
            _UNLOCK_CYCLES,
        ),
        "unlock_shield_generator": UnlockRule(
            ("shield_generator",), _UNLOCK_COST, _UNLOCK_CYCLES
        ),
        "unlock_targeting_computer": UnlockRule(
            ("targeting_computer",), _UNLOCK_COST, _UNLOCK_CYCLES
        ),
        "unlock_df_antenna": UnlockRule(("df_antenna",), _UNLOCK_COST, _UNLOCK_CYCLES),
        "unlock_jammer": UnlockRule(("jammer",), _UNLOCK_COST, _UNLOCK_CYCLES),
        "unlock_seismic_sensor": UnlockRule(
            ("seismic_sensor",), _UNLOCK_COST, _UNLOCK_CYCLES
        ),
        "unlock_land_mine": UnlockRule(("land_mine",), _UNLOCK_COST, _UNLOCK_CYCLES),
        "unlock_signature_dampener": UnlockRule(
            ("signature_dampener",), _UNLOCK_COST, _UNLOCK_CYCLES
        ),
        "unlock_decoy_beacon": UnlockRule(
            ("decoy_beacon",), _UNLOCK_COST, _UNLOCK_CYCLES
        ),
        "unlock_smoke_launcher": UnlockRule(
            ("smoke_launcher",), _UNLOCK_COST, _UNLOCK_CYCLES
        ),
        "unlock_point_defense": UnlockRule(
            ("point_defense",), _UNLOCK_COST, _UNLOCK_CYCLES
        ),
        "unlock_howitzer": UnlockRule(
            ("howitzer", "howitzer_shells", "howitzer_charges"),
            _UNLOCK_COST,
            _UNLOCK_CYCLES,
        ),
        "unlock_antenna": UnlockRule(("antenna",), _UNLOCK_COST, _UNLOCK_CYCLES),
        "unlock_repeater": UnlockRule(("repeater",), _UNLOCK_COST, _UNLOCK_CYCLES),
    }
)


# Equipment upgrade tiers are common to every ordinary equipment research
# path.  Baseline chassis research is the documented exception.
UPGRADE_TIER_RULES: Mapping[int, TimedCost] = MappingProxyType(
    {
        1: TimedCost(_cost(enriched_unobtanium=1), 100),
        2: TimedCost(_cost(enriched_unobtanium=3), 200),
        3: TimedCost(_cost(enriched_unobtanium=5), 300),
    }
)

BASELINE_UPGRADE_TIER_RULES: Mapping[int, TimedCost] = MappingProxyType(
    {
        1: TimedCost(_cost(enriched_unobtanium=3), 150),
        2: TimedCost(_cost(enriched_unobtanium=4), 300),
        3: TimedCost(_cost(enriched_unobtanium=6), 300),
    }
)


REFINING_RULES: Mapping[str, RefiningRule] = MappingProxyType(
    {
        "titanium_ore": RefiningRule("titanium_parts", 0.50),
        "rare_earth_ore": RefiningRule("battery_materials", 0.25),
        "unobtanium_ore": RefiningRule("enriched_unobtanium", 0.02),
    }
)


def building_rule(building_type: str) -> BuildingRule:
    """Return construction and timer data for a buildable building."""

    return BUILDING_RULES[building_type]


def building_action_cycles(building_type: str, action: str) -> int:
    """Return an action's base cycles, including common building actions."""

    rule = building_rule(building_type)
    action = _BUILDING_ACTION_ALIASES.get(action, action)
    try:
        return rule.action_cycles[action]
    except KeyError:
        return _COMMON_BUILDING_ACTION_CYCLES[action]


def unlock_rule(research_key: str) -> UnlockRule:
    """Return the cost, timer, and products for an unlock research key."""

    return UNLOCK_RULES[research_key]


def upgrade_tier_rule(tier: int, *, baseline_drone: bool = False) -> TimedCost:
    """Return cost and timer for an equipment or baseline chassis tier."""

    rules = BASELINE_UPGRADE_TIER_RULES if baseline_drone else UPGRADE_TIER_RULES
    return rules[tier]


def refining_rule(ore_type: str) -> RefiningRule:
    """Return output, probability, and per-unit timer for an ore type."""

    return REFINING_RULES[ore_type]
