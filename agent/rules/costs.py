"""Static game costs and pure cost calculations.

The values in this module are ported from the seed client's ``map_store.py``
and ``message_parser.py``.  They intentionally preserve the seed behaviour,
including values that disagree with ``api_reference_old.md`` or
``game_guide_old.md``.  Those disagreements are listed in
``CALIBRATION_TODOS`` and must be measured against a private server in E4
before changing the operational tables.

There is no I/O or mutable runtime state here.  Callers supply research tiers,
weight, and action details explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

ResourceCost = dict[str, int]
BatteryCostSpec = tuple[dict[int, int], str | None, float]


# Equipment base weights and per-tier research modifiers from map_store.py.
EQUIPMENT_WEIGHTS: dict[str, int] = {
    "propulsion": 30,
    "hover_tech": 25,
    "laser_cannon": 20,
    "wire_guided_missile": 15,
    "missile_launcher": 15,
    "targeting_computer": 10,
    "reactive_armour": 120,
    "auto_cannon": 20,
    "ammunition": 10,
    "armour_piercing_ammunition": 10,
    "high_explosive_ammunition": 10,
    "canister_round_ammunition": 10,
    "shield_generator": 25,
    "df_antenna": 30,
    "jammer": 30,
    "antenna": 10,
    "repeater": 25,
    "land_mine": 15,
    "mine_layer": 15,
    "signature_dampener": 15,
    "decoy_beacon": 15,
    "smoke_launcher": 15,
    "point_defense": 25,
    "seismic_sensor": 10,
    "howitzer": 40,
    "howitzer_shells": 25,
    "howitzer_charges": 20,
    "sensors": 10,
    "drill": 25,
    "hopper": 15,
    "battery": 30,
    "portable_charger": 20,
    "mobile_repair_kit": 20,
}

EQUIPMENT_WEIGHT_RESEARCH: dict[str, int] = {
    "reactive_armour": -20,
    "ammunition": -3,
    "canister_round": -3,
    "howitzer_shells": -5,
    "howitzer_charges": -5,
    "baseline_drone": 10,
}


# equipment_type -> (base count, per-tier bonus, research key)
CONSUMABLE_SPECS: dict[str, tuple[int, int, str]] = {
    "ammunition": (10, 3, "ammunition"),
    "armour_piercing_ammunition": (10, 3, "ammunition"),
    "high_explosive_ammunition": (10, 3, "ammunition"),
    "canister_round_ammunition": (10, 3, "canister_round"),
    "wire_guided_missile": (2, 0, "wire_guided_missile"),
    "missile_launcher": (2, 0, "wire_guided_missile"),
    "land_mine": (3, 1, "land_mine"),
    "mine_layer": (3, 1, "land_mine"),
    "decoy_beacon": (3, 1, "decoy_beacon"),
    "smoke_launcher": (3, 1, "smoke_launcher"),
    "howitzer_shells": (5, 1, "howitzer_shells"),
    "howitzer_charges": (15, 3, "howitzer_charges"),
}


BUILDING_RESOURCE_CAPACITY: dict[str, int] = {
    "silo": 500,
    "command_center": 400,
    "refiner": 200,
    "drone_production_plant": 100,
    "drone_equipment_production_plant": 200,
    "laser_turret": 100,
    "missile_turret": 100,
    "auto_cannon_turret": 100,
    "lucas_launcher": 180,
}

CHARGE_RATE = 250


# Known resource costs from message_parser.py.
EQUIPMENT_COSTS: dict[str, ResourceCost] = {
    "drill": {"titanium_parts": 8, "battery_materials": 4},
    "sensors": {"titanium_parts": 4, "battery_materials": 8},
    "battery": {"titanium_parts": 6, "battery_materials": 14},
    "portable_charger": {"titanium_parts": 6, "battery_materials": 10},
    "reactive_armour": {"titanium_parts": 22, "battery_materials": 4},
    "hopper": {"titanium_parts": 6, "battery_materials": 2},
    "mobile_repair_kit": {"titanium_parts": 10, "battery_materials": 6},
    "shield_generator": {"titanium_parts": 8, "battery_materials": 12},
    "laser_cannon": {"titanium_parts": 18, "battery_materials": 12},
    "propulsion": {"titanium_parts": 14, "battery_materials": 6},
    "hover_tech": {"titanium_parts": 14, "battery_materials": 8},
    "targeting_computer": {"titanium_parts": 4, "battery_materials": 8},
    "wire_guided_missile": {"titanium_parts": 24, "battery_materials": 12},
    "auto_cannon": {"titanium_parts": 20, "battery_materials": 10},
    "armour_piercing_ammunition": {"titanium_parts": 8, "battery_materials": 2},
    "high_explosive_ammunition": {"titanium_parts": 8, "battery_materials": 2},
    "canister_round_ammunition": {"titanium_parts": 8, "battery_materials": 2},
    "antenna": {"titanium_parts": 10, "battery_materials": 12},
    "repeater": {"titanium_parts": 18, "battery_materials": 16},
}

DRONE_COST: ResourceCost = {"titanium_parts": 30, "battery_materials": 15}
LUCAS_COST: ResourceCost = {"titanium_parts": 30, "battery_materials": 15}
MISSILE_RELOAD_COST: ResourceCost = {"titanium_parts": 12, "battery_materials": 8}
AUTO_CANNON_RELOAD_COST: ResourceCost = {"titanium_parts": 6, "battery_materials": 2}

BUILDING_COSTS: dict[str, ResourceCost] = {
    "solar_panel_array": {"titanium_parts": 40, "battery_materials": 20},
    "battery_array": {"titanium_parts": 40, "battery_materials": 30},
    "drone_charging_station": {"titanium_parts": 50, "battery_materials": 20},
    "drone_repair_station": {"titanium_parts": 50, "battery_materials": 25},
    "silo": {"titanium_parts": 20, "battery_materials": 5},
    "refiner": {"titanium_parts": 100, "battery_materials": 40},
    "drone_production_plant": {"titanium_parts": 120, "battery_materials": 50},
    "drone_equipment_production_plant": {
        "titanium_parts": 140,
        "battery_materials": 60,
    },
    "shield_array": {
        "titanium_parts": 100,
        "battery_materials": 50,
        "enriched_unobtanium": 2,
    },
    "laser_turret": {
        "titanium_parts": 100,
        "battery_materials": 40,
        "enriched_unobtanium": 1,
    },
    "missile_turret": {
        "titanium_parts": 80,
        "battery_materials": 30,
        "enriched_unobtanium": 1,
    },
    "auto_cannon_turret": {
        "titanium_parts": 90,
        "battery_materials": 35,
        "enriched_unobtanium": 1,
    },
    "sensor_array": {
        "titanium_parts": 60,
        "battery_materials": 40,
        "enriched_unobtanium": 1,
    },
    "point_defense_system": {
        "titanium_parts": 80,
        "battery_materials": 50,
        "enriched_unobtanium": 2,
    },
    "lucas_launcher": {
        "titanium_parts": 120,
        "battery_materials": 50,
        "enriched_unobtanium": 5,
    },
}

REPAIR_COST: ResourceCost = {"titanium_parts": 3, "battery_materials": 1}
UNLOCK_COST: ResourceCost = {"enriched_unobtanium": 4}
UPGRADE_TIER_COST: dict[int, int] = {1: 1, 2: 3, 3: 5}
BASELINE_TIER_COST: dict[int, int] = {1: 3, 2: 4, 3: 6}


# Entity-effect specs from message_parser.py, colocated with their costs so the
# local entity simulator does not need to duplicate game constants.
REPAIR_KIT_HP: dict[int, int] = {1: 5, 2: 15, 3: 30}
REPAIR_KIT_DURABILITY: dict[int, int] = {1: 25, 2: 50, 3: 75}
REPAIR_STATION_HP = 60
REPAIR_STATION_DURABILITY = 50
SHIELD_CHARGE_RATE: dict[int, int] = {1: 5, 2: 15, 3: 30}
SHIELD_GENERATE_RATE = 25
SHIELD_BUFFER_CAPACITY = 500

# level -> (transfer rate, base efficiency); research adds +10 and +0.1/tier.
CHARGER_SPECS: dict[int, tuple[int, float]] = {
    1: (10, 1.0),
    2: (25, 0.8),
    3: (50, 0.6),
}


# action-completed subject fragment -> (level costs, research key, reduction/tier)
BATTERY_COSTS: dict[str, BatteryCostSpec] = {
    "drive completed": ({1: 5, 2: 10, 3: 20}, "propulsion", 0.15),
    "turn completed": ({1: 2, 2: 5, 3: 10}, "propulsion", 0.15),
    "scan completed": ({1: 1, 2: 5, 3: 10}, "sensors", 0.15),
    "identify completed": ({1: 1, 2: 5, 3: 10}, "sensors", 0.15),
    "mine completed": ({1: 2, 2: 5, 3: 10}, "drill", 0.10),
    "fire completed": ({1: 10, 2: 20, 3: 30}, None, 0),
    "fire cannon completed": ({1: 10, 2: 25, 3: 45}, None, 0),
    "fire missile completed": ({1: 10, 2: 20, 3: 30}, None, 0),
    "autocannon load completed": ({1: 20}, None, 0),
    "autocannon unload completed": ({1: 10}, None, 0),
    "charge shield completed": ({1: 15, 2: 30, 3: 40}, "shield_generator", 0.10),
    "transfer shield completed": ({1: 5, 2: 10, 3: 15}, None, 0),
    "mark completed": ({1: 5, 2: 15, 3: 30}, "targeting_computer", 0.10),
    "charge completed": ({1: 50, 2: 67, 3: 100}, None, 0),
    "repair completed": ({1: 50, 2: 75, 3: 100}, "mobile_repair_kit", 0.10),
    "repair equipment completed": ({1: 10, 2: 35, 3: 60}, "mobile_repair_kit", 0.10),
    "dump hopper completed": ({1: 1}, None, 0),
    "land mine deploy completed": ({1: 20, 2: 45, 3: 50}, None, 0),
    "decoy deploy completed": ({1: 20, 2: 60, 3: 50}, None, 0),
    "smoke deploy completed": ({1: 10, 2: 30, 3: 30}, None, 0),
    "hover ascend completed": ({1: 5, 2: 12, 3: 25}, "hover_tech", 0.15),
    "hover descend completed": ({1: 3, 2: 8, 3: 18}, "hover_tech", 0.15),
    "howitzer deploy completed": ({1: 8, 2: 10, 3: 21}, "howitzer", 0.10),
    "howitzer stow completed": ({1: 8, 2: 10, 3: 21}, "howitzer", 0.10),
    "howitzer load shell completed": ({1: 8, 2: 10, 3: 21}, "howitzer", 0.10),
    "howitzer load charges completed": ({1: 8, 2: 10, 3: 21}, "howitzer", 0.10),
    "howitzer aim completed": ({1: 2, 2: 3, 3: 8}, "howitzer", 0.10),
    "howitzer fire completed": ({1: 8, 2: 10, 3: 21}, "howitzer", 0.10),
    "df deploy completed": ({1: 10}, None, 0),
    "df stow completed": ({1: 10}, None, 0),
    "df pack completed": ({1: 10}, None, 0),
    "df raise completed": ({1: 2}, None, 0),
    "df lower completed": ({1: 2}, None, 0),
    "df configure completed": ({1: 1}, None, 0),
    "jammer deploy completed": ({1: 10}, None, 0),
    "jammer stow completed": ({1: 10}, None, 0),
    "jammer pack completed": ({1: 10}, None, 0),
    "jammer raise completed": ({1: 2}, None, 0),
    "jammer lower completed": ({1: 2}, None, 0),
    "jammer activate completed": ({1: 5}, None, 0),
    "jammer deactivate completed": ({1: 0}, None, 0),
    "repeater deploy completed": ({1: 10}, None, 0),
    "repeater stow completed": ({1: 10}, None, 0),
    "repeater pack completed": ({1: 10}, None, 0),
    "repeater raise completed": ({1: 2}, None, 0),
    "repeater lower completed": ({1: 2}, None, 0),
    "repeater activate completed": ({1: 5}, None, 0),
    "repeater deactivate completed": ({1: 0}, None, 0),
    "point defense activate": ({1: 5, 2: 10, 3: 30}, None, 0),
    "status completed": ({1: 1}, None, 0),
    "ammo transfer completed": ({1: 15, 2: 20, 3: 30}, None, 0),
}

# Retain the seed name for code comparing the port directly with message_parser.
_BATTERY_COSTS = BATTERY_COSTS


# E4 CALIBRATION TODOs: known old-document versus seed-code conflicts.
CALIBRATION_TODOS: tuple[str, ...] = (
    "E4: measure command_center resource capacity (seed table: 400; old game guide: 2000).",
    "E4: measure HE ammunition manufacture cost (seed: 8 titanium/2 battery; guide: 4/6).",
    "E4: measure canister ammunition manufacture cost (seed: 8 titanium/2 battery; guide: 6/4).",
    "E4: calibrate equipment-cost coverage: the guide publishes df_antenna, jammer, land_mine, "
    "signature_dampener, decoy_beacon, smoke_launcher, point_defense, seismic_sensor, and "
    "howitzer-family costs missing from the seed table; seed-only antenna/repeater names also "
    "need mapping to server equipment types.",
    "E4: measure antenna/repeater equipment weights (seed: 10/25); the old reference and "
    "guide list the server types but do not publish their weights.",
    "E4: measure autocannon load/unload battery (seed: 20/10; guide says load is 0 and does "
    "not publish unload).",
    "E4: measure dump-hopper battery (seed: 1; guide: 0).",
    "E4: measure shield-charge battery by level (seed total: 15/30/40; guide: 5/20/40).",
    "E4: measure mobile drone-repair battery by level (seed: 50/75/100; guide: 5/25/60).",
    "E4: measure DF mast action battery (seed deploy/pack 10, raise/lower 2 per 5 units, "
    "configure 1; guide lists all as 0).",
    "E4: measure jammer lifecycle battery (seed deploy/pack 10, raise/lower 2 per 5 units, "
    "activate 5; guide lists lifecycle actions as 0 and only documents active drain).",
    "E4: measure all repeater lifecycle costs; the seed mirrors jammer values but the old "
    "reference and guide do not publish repeater costs.",
    "E4: verify hover weight surcharge; the guide says hover costs scale with weight while the "
    "seed estimator applies a weight surcharge only to drive.",
    "E4: verify dual-propulsion cost; the guide specifies 1.5x base cost while the seed "
    "estimator has no propulsion-count input.",
    "E4: calibrate point-defense totals for cycles and successful interceptions; the seed "
    "estimator charges one activation only, while the guide adds per-cycle and per-intercept cost.",
)


def consumable_capacity(
    equipment_type: str, research_tiers: Mapping[str, int] | None = None
) -> int | None:
    """Return a consumable's maximum count after research, or ``None`` if not consumable."""

    spec = CONSUMABLE_SPECS.get(equipment_type)
    if spec is None:
        return None
    base, per_tier, research_key = spec
    tier = (research_tiers or {}).get(research_key, 0)
    return base + per_tier * tier


def estimate_battery_cost(
    subject: str,
    details: Mapping[str, object] | None = None,
    *,
    research_tiers: Mapping[str, int] | None = None,
    total_weight: float = 0,
) -> int:
    """Estimate battery spent by a completed drone action.

    This is the seed client's ``_estimate_battery_cost`` made pure: instead of
    looking up a drone and team research in ``MapStore``, the caller supplies
    those values.  ``details`` may contain ``level``, howitzer aim coordinates,
    or mast/elevation change.  Unknown actions cost zero and unknown levels
    fall back to level 1, matching the seed.
    """

    action_details = details or {}
    level = int(action_details.get("level", 1))
    subject_lower = subject.lower()

    # Match specific subjects before generic ones.  The seed iterates insertion
    # order, which makes "fire completed" shadow "howitzer fire completed"
    # despite its own table and the guide both specifying howitzer costs.
    action_keys = sorted(BATTERY_COSTS, key=len, reverse=True)
    for action_key in action_keys:
        if action_key not in subject_lower:
            continue

        costs, research_key, reduction = BATTERY_COSTS[action_key]
        base = costs.get(level, costs.get(1, 0))
        if base == 0:
            return 0

        if research_key and reduction > 0:
            tier = (research_tiers or {}).get(research_key, 0)
            base *= 1 - reduction * tier

        cost = max(1, int(base))

        if "drive completed" in subject_lower and total_weight > 0:
            cost += math.ceil(total_weight / 20)

        if "howitzer aim completed" in subject_lower:
            target_q = action_details.get("target_q")
            target_r = action_details.get("target_r")
            if target_q is not None and target_r is not None:
                distance = _hex_distance_from_origin(int(target_q), int(target_r))
                cost = max(1, int(base * distance))

        if "raise completed" in subject_lower or "lower completed" in subject_lower:
            height_change = action_details.get("height_change") or action_details.get(
                "elevation_change", 5
            )
            units = max(1, int(height_change) // 5)
            cost = max(1, int(base * units))

        return cost

    return 0


def _hex_distance_from_origin(q: int, r: int) -> int:
    """Odd-q distance from (0, 0), matching the seed hex helper."""

    x = q
    z = r - (q - (q & 1)) // 2
    y = -x - z
    return max(abs(x), abs(y), abs(z))
