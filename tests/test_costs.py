"""Golden tests for the seed-client cost/spec port."""

from __future__ import annotations

import pytest

from agent.rules.costs import (
    AUTO_CANNON_RELOAD_COST,
    BATTERY_COSTS,
    BUILDING_COSTS,
    BUILDING_RESOURCE_CAPACITY,
    CALIBRATION_TODOS,
    CHARGE_RATE,
    CONSUMABLE_SPECS,
    DRONE_COST,
    EQUIPMENT_COSTS,
    EQUIPMENT_WEIGHTS,
    MISSILE_RELOAD_COST,
    consumable_capacity,
    estimate_battery_cost,
)


GOLDEN_BATTERY_COSTS = {
    "drive completed": {1: 5, 2: 10, 3: 20},
    "turn completed": {1: 2, 2: 5, 3: 10},
    "scan completed": {1: 1, 2: 5, 3: 10},
    "identify completed": {1: 1, 2: 5, 3: 10},
    "mine completed": {1: 2, 2: 5, 3: 10},
    "fire completed": {1: 10, 2: 20, 3: 30},
    "fire cannon completed": {1: 10, 2: 25, 3: 45},
    "fire missile completed": {1: 10, 2: 20, 3: 30},
    "autocannon load completed": {1: 20},
    "autocannon unload completed": {1: 10},
    "charge shield completed": {1: 15, 2: 30, 3: 40},
    "transfer shield completed": {1: 5, 2: 10, 3: 15},
    "mark completed": {1: 5, 2: 15, 3: 30},
    "charge completed": {1: 50, 2: 67, 3: 100},
    "repair completed": {1: 50, 2: 75, 3: 100},
    "repair equipment completed": {1: 10, 2: 35, 3: 60},
    "dump hopper completed": {1: 1},
    "land mine deploy completed": {1: 20, 2: 45, 3: 50},
    "decoy deploy completed": {1: 20, 2: 60, 3: 50},
    "smoke deploy completed": {1: 10, 2: 30, 3: 30},
    "hover ascend completed": {1: 5, 2: 12, 3: 25},
    "hover descend completed": {1: 3, 2: 8, 3: 18},
    "howitzer deploy completed": {1: 8, 2: 10, 3: 21},
    "howitzer stow completed": {1: 8, 2: 10, 3: 21},
    "howitzer load shell completed": {1: 8, 2: 10, 3: 21},
    "howitzer load charges completed": {1: 8, 2: 10, 3: 21},
    "howitzer aim completed": {1: 2, 2: 3, 3: 8},
    "howitzer fire completed": {1: 8, 2: 10, 3: 21},
    "df deploy completed": {1: 10},
    "df stow completed": {1: 10},
    "df pack completed": {1: 10},
    "df raise completed": {1: 2},
    "df lower completed": {1: 2},
    "df configure completed": {1: 1},
    "jammer deploy completed": {1: 10},
    "jammer stow completed": {1: 10},
    "jammer pack completed": {1: 10},
    "jammer raise completed": {1: 2},
    "jammer lower completed": {1: 2},
    "jammer activate completed": {1: 5},
    "jammer deactivate completed": {1: 0},
    "repeater deploy completed": {1: 10},
    "repeater stow completed": {1: 10},
    "repeater pack completed": {1: 10},
    "repeater raise completed": {1: 2},
    "repeater lower completed": {1: 2},
    "repeater activate completed": {1: 5},
    "repeater deactivate completed": {1: 0},
    "point defense activate": {1: 5, 2: 10, 3: 30},
    "status completed": {1: 1},
    "ammo transfer completed": {1: 15, 2: 20, 3: 30},
}


@pytest.mark.parametrize(
    ("subject", "level", "expected"),
    [
        (subject, level, cost)
        for subject, level_costs in GOLDEN_BATTERY_COSTS.items()
        for level, cost in level_costs.items()
    ],
)
def test_battery_cost_golden_per_action(subject: str, level: int, expected: int) -> None:
    assert estimate_battery_cost(subject, {"level": level}) == expected


def test_battery_table_has_exact_seed_action_coverage() -> None:
    assert set(BATTERY_COSTS) == set(GOLDEN_BATTERY_COSTS)
    assert {key: value[0] for key, value in BATTERY_COSTS.items()} == GOLDEN_BATTERY_COSTS


def test_battery_research_reduction_truncates_like_seed() -> None:
    assert estimate_battery_cost(
        "Drive completed", {"level": 3}, research_tiers={"propulsion": 2}
    ) == 14
    assert estimate_battery_cost(
        "Howitzer Fire completed", {"level": 3}, research_tiers={"howitzer": 3}
    ) == 14


def test_drive_adds_ceiling_weight_surcharge_after_research() -> None:
    assert estimate_battery_cost(
        "Drive completed",
        {"level": 2},
        research_tiers={"propulsion": 1},
        total_weight=101,
    ) == 14  # int(10 * .85) + ceil(101 / 20)


def test_howitzer_aim_multiplies_cost_by_odd_q_hex_distance() -> None:
    assert estimate_battery_cost(
        "Howitzer Aim completed",
        {"level": 2, "target_q": 3, "target_r": 2},
    ) == 12  # distance 4 * 3 battery/tile


@pytest.mark.parametrize("subject", ["DF Raise completed", "Jammer Lower completed"])
def test_mast_cost_scales_per_five_height_units(subject: str) -> None:
    assert estimate_battery_cost(subject, {"height_change": 15}) == 6
    assert estimate_battery_cost(subject, {"elevation_change": 10}) == 4


def test_unknown_action_is_free_and_unknown_level_falls_back_to_one() -> None:
    assert estimate_battery_cost("Celebrate completed", {"level": 3}) == 0
    assert estimate_battery_cost("SCAN COMPLETED", {"level": 99}) == 1


def test_consumable_capacity_uses_shared_research_key() -> None:
    assert consumable_capacity("armour_piercing_ammunition", {"ammunition": 2}) == 16
    assert consumable_capacity("canister_round_ammunition", {"canister_round": 3}) == 19
    assert consumable_capacity("wire_guided_missile", {"wire_guided_missile": 3}) == 2
    assert consumable_capacity("laser_cannon") is None


def test_map_store_spec_tables_match_seed_golden_values() -> None:
    assert EQUIPMENT_WEIGHTS["reactive_armour"] == 120
    assert EQUIPMENT_WEIGHTS["howitzer"] == 40
    assert CONSUMABLE_SPECS["land_mine"] == (3, 1, "land_mine")
    assert CONSUMABLE_SPECS["howitzer_charges"] == (15, 3, "howitzer_charges")
    assert BUILDING_RESOURCE_CAPACITY == {
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
    assert CHARGE_RATE == 250


def test_resource_cost_golden_values() -> None:
    assert EQUIPMENT_COSTS["laser_cannon"] == {
        "titanium_parts": 18,
        "battery_materials": 12,
    }
    assert DRONE_COST == {"titanium_parts": 30, "battery_materials": 15}
    assert MISSILE_RELOAD_COST == {"titanium_parts": 12, "battery_materials": 8}
    assert AUTO_CANNON_RELOAD_COST == {"titanium_parts": 6, "battery_materials": 2}
    assert BUILDING_COSTS["lucas_launcher"] == {
        "titanium_parts": 120,
        "battery_materials": 50,
        "enriched_unobtanium": 5,
    }


def test_all_doc_code_conflicts_are_tagged_for_e4_calibration() -> None:
    assert len(CALIBRATION_TODOS) == 15
    assert all(todo.startswith("E4:") for todo in CALIBRATION_TODOS)
