"""Golden tests for economy, construction, and research rule data."""

from __future__ import annotations

import pytest

from agent.rules.costs import BUILDING_COSTS
from agent.rules.economy import (
    BASELINE_UPGRADE_TIER_RULES,
    BUILDING_RULES,
    REFINING_RULES,
    UNLOCK_RULES,
    UPGRADE_TIER_RULES,
    building_action_cycles,
    building_rule,
    refining_rule,
    unlock_rule,
    upgrade_tier_rule,
)


EXPECTED_UNLOCKS = {
    "unlock_reactive_armour",
    "unlock_hover_tech",
    "unlock_wire_guided_missile",
    "unlock_auto_cannon",
    "unlock_shield_generator",
    "unlock_targeting_computer",
    "unlock_df_antenna",
    "unlock_jammer",
    "unlock_seismic_sensor",
    "unlock_land_mine",
    "unlock_signature_dampener",
    "unlock_decoy_beacon",
    "unlock_smoke_launcher",
    "unlock_point_defense",
    "unlock_howitzer",
    "unlock_antenna",
    "unlock_repeater",
}


def test_every_buildable_building_has_its_construction_cost_and_timers() -> None:
    assert set(BUILDING_RULES) == set(BUILDING_COSTS)
    for building_type, expected_cost in BUILDING_COSTS.items():
        rule = building_rule(building_type)
        assert dict(rule.cost) == expected_cost
        assert rule.action_cycles is not None


@pytest.mark.parametrize(
    ("building_type", "action", "expected"),
    [
        ("solar_panel_array", "reposition", 1),
        ("drone_charging_station", "charge", 10),
        ("drone_production_plant", "manufacture", 500),
        ("drone_production_plant", "build_drone", 500),
        ("drone_equipment_production_plant", "manufacture", 25),
        ("auto_cannon_turret", "fire_cannon", 1),
        ("missile_turret", "reload", 40),
        ("lucas_launcher", "launch", 25),
        ("drone_repair_station", "repair", 5),
        ("refiner", "repair", 20),
        ("battery_array", "status", 1),
    ],
)
def test_building_action_timer_queries(
    building_type: str, action: str, expected: int
) -> None:
    assert building_action_cycles(building_type, action) == expected


def test_unlock_tree_has_all_17_single_tier_research_keys() -> None:
    assert set(UNLOCK_RULES) == EXPECTED_UNLOCKS
    assert len(UNLOCK_RULES) == 17
    for key in EXPECTED_UNLOCKS:
        rule = unlock_rule(key)
        assert dict(rule.cost) == {"enriched_unobtanium": 4}
        assert rule.cycles == 400


def test_grouped_unlocks_list_every_product() -> None:
    assert unlock_rule("unlock_auto_cannon").unlocks == (
        "auto_cannon",
        "armour_piercing_ammunition",
        "high_explosive_ammunition",
        "canister_round_ammunition",
    )
    assert unlock_rule("unlock_howitzer").unlocks == (
        "howitzer",
        "howitzer_shells",
        "howitzer_charges",
    )


def test_upgrade_tier_costs_and_timers() -> None:
    assert set(UPGRADE_TIER_RULES) == {1, 2, 3}
    costs = [
        upgrade_tier_rule(tier).cost["enriched_unobtanium"] for tier in (1, 2, 3)
    ]
    assert costs == [1, 3, 5]
    assert [upgrade_tier_rule(tier).cycles for tier in (1, 2, 3)] == [100, 200, 300]


def test_baseline_chassis_uses_its_exceptional_costs_and_timers() -> None:
    assert set(BASELINE_UPGRADE_TIER_RULES) == {1, 2, 3}
    assert [
        upgrade_tier_rule(tier, baseline_drone=True).cost["enriched_unobtanium"]
        for tier in (1, 2, 3)
    ] == [3, 4, 6]
    assert [
        upgrade_tier_rule(tier, baseline_drone=True).cycles for tier in (1, 2, 3)
    ] == [150, 300, 300]


def test_refining_probabilities_outputs_and_per_unit_timers() -> None:
    assert set(REFINING_RULES) == {
        "titanium_ore",
        "rare_earth_ore",
        "unobtanium_ore",
    }
    assert refining_rule("titanium_ore").output_resource == "titanium_parts"
    assert refining_rule("titanium_ore").success_probability == 0.50
    assert refining_rule("rare_earth_ore").output_resource == "battery_materials"
    assert refining_rule("rare_earth_ore").success_probability == 0.25
    assert refining_rule("unobtanium_ore").output_resource == "enriched_unobtanium"
    assert refining_rule("unobtanium_ore").success_probability == 0.02
    assert all(rule.cycles_per_unit == 1 for rule in REFINING_RULES.values())


@pytest.mark.parametrize(
    ("query", "argument"),
    [
        (building_rule, "orbital_shipyard"),
        (unlock_rule, "unlock_teleporter"),
        (upgrade_tier_rule, 4),
        (refining_rule, "moon_rock"),
    ],
)
def test_unknown_queries_raise_key_error(query: object, argument: object) -> None:
    with pytest.raises(KeyError):
        query(argument)  # type: ignore[operator]


def test_published_rule_tables_are_immutable() -> None:
    with pytest.raises(TypeError):
        BUILDING_RULES["silo"] = BUILDING_RULES["silo"]  # type: ignore[index]
    with pytest.raises(TypeError):
        unlock_rule("unlock_antenna").cost["enriched_unobtanium"] = 99  # type: ignore[index]
