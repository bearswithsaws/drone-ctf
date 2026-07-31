"""Replay and accounting tests for the local entity simulator."""

from __future__ import annotations

from agent.bus import EventBus
from agent.sim.entity_sim import EntitySim
from agent.transport.parsing import parse_wire_message
from agent.transport.ws import TOPIC_MESSAGE


def queued(
    action_id: str,
    subject: str,
    details: dict[str, object],
    *,
    drone_id: str = "drone-1",
) -> dict[str, object]:
    return {
        "message_type": "action_queued",
        "action_id": action_id,
        "drone_id": drone_id,
        "subject": subject,
        "details": details,
    }


def completed(
    action_id: str,
    subject: str,
    details: dict[str, object] | None = None,
    *,
    drone_id: str = "drone-1",
) -> dict[str, object]:
    return {
        "message_type": "action_completed",
        "action_id": action_id,
        "drone_id": drone_id,
        "subject": subject,
        "details": {"success": True, **(details or {})},
    }


def building_completed(
    action_id: str,
    building_id: str,
    subject: str,
    details: dict[str, object] | None = None,
    *,
    drone_id: str | None = None,
) -> dict[str, object]:
    message: dict[str, object] = {
        "message_type": "building_action_completed",
        "action_id": action_id,
        "building_id": building_id,
        "subject": subject,
        "details": {"success": True, **(details or {})},
    }
    if drone_id is not None:
        message["drone_id"] = drone_id
    return message


def test_replay_uses_queued_level_weight_and_research_for_battery() -> None:
    sim = EntitySim(research_tiers={"propulsion": 1})
    sim.seed_drone(
        "drone-1",
        current_battery=100,
        max_battery=100,
        total_weight=41,
    )

    records = [
        {"topic": "message", "payload": queued("a1", "Drive queued", {"level": 2})},
        {"topic": "action.submitted", "payload": {"ignored": True}},
        {"topic": "message", "payload": completed("a1", "Drive completed")},
        {"topic": "message", "payload": queued("a2", "Turn queued", {"level": 3})},
        {"topic": "message", "payload": completed("a2", "Turn completed")},
    ]

    deltas = sim.replay(records)

    # Drive: int(10 * .85) + ceil(41/20) = 11. Turn: int(10 * .85) = 8.
    assert [delta.battery_spent for delta in deltas] == [11, 8]
    assert sim.get_drone("drone-1").current_battery == 81  # type: ignore[union-attr]


def test_status_is_authoritative_and_extracts_all_stripped_fields() -> None:
    sim = EntitySim()
    message = completed(
        "status-1",
        "Status completed",
        {
            "status": {
                "drone_id": "drone-1",
                "current_battery": 3999,
                "max_battery": 4000,
                "chassis_weight": 20,
                "equipment": [
                    {"equipment_type": "propulsion", "weight": 30},
                    {
                        "equipment_type": "armour_piercing_ammunition",
                        "weight": 10,
                        "rounds_remaining": 7,
                    },
                    {
                        "equipment_type": "land_mine",
                        "weight": 15,
                        "mines_remaining": 2,
                    },
                    {
                        "equipment_type": "decoy_beacon",
                        "weight": 15,
                        "decoys_remaining": 3,
                    },
                    {
                        "equipment_type": "howitzer_charges",
                        "weight": 20,
                        "charges_remaining": 11,
                    },
                    {
                        "equipment_type": "auto_cannon",
                        "weight": 20,
                        "loaded_ammo_type": "armour_piercing",
                    },
                    {
                        "equipment_type": "hopper",
                        "weight": 15,
                        "capacity": 25,
                        "current_load": {"titanium_ore": 12},
                    },
                ],
            }
        },
    )

    delta = sim.apply_message(message)
    state = sim.get_drone("drone-1")

    assert delta is not None and delta.status_checkpoint
    assert delta.battery_spent == 0  # the reported 3999 already includes status's cost
    assert state is not None
    assert state.current_battery == 3999
    assert state.total_weight == 145
    assert state.loaded_ammo_type == "armour_piercing_ammunition"
    assert state.consumables == {
        "armour_piercing_ammunition": 7,
        "land_mine": 2,
        "decoy_beacon": 3,
        "howitzer_charges": 11,
    }
    assert state.cargo == {"titanium_ore": 12}
    assert state.hopper_load == 12
    assert state.hopper_capacity == 25


def test_mine_unload_and_station_charge_update_drone_projection() -> None:
    sim = EntitySim()
    sim.seed_drone(
        "drone-1",
        current_battery=100,
        max_battery=500,
        cargo={"titanium_ore": 5},
        hopper_capacity=25,
    )
    sim.seed_building("refiner", building_type="refiner")
    sim.seed_building("charger", building_type="drone_charging_station")

    sim.apply_message(
        completed(
            "mine-1",
            "Mine completed",
            {"level": 1, "total_ore_mined": 20},
        )
    )
    mined = sim.get_drone("drone-1")
    assert mined is not None
    assert mined.current_battery == 98
    assert mined.hopper_load == 25

    sim.apply_message(
        building_completed(
            "unload-1",
            "refiner",
            "Unload completed",
            {"ore_unloaded": {"titanium_ore": 25}},
            drone_id="drone-1",
        )
    )
    unloaded = sim.get_drone("drone-1")
    refiner = sim.get_building("refiner")
    assert unloaded is not None and unloaded.cargo == {}
    assert refiner is not None
    assert refiner.stored_resources == {"titanium_ore": 25}

    sim.apply_message(
        building_completed(
            "charge-1",
            "charger",
            "Charge completed",
            drone_id="drone-1",
        )
    )
    charged = sim.get_drone("drone-1")
    assert charged is not None and charged.current_battery == 348


def test_consumables_apply_only_on_completion_and_never_go_negative() -> None:
    sim = EntitySim()
    sim.seed_drone(
        "drone-1",
        current_battery=1000,
        max_battery=1000,
        consumables={
            "wire_guided_missile": 1,
            "land_mine": 1,
            "decoy_beacon": 1,
            "smoke_launcher": 1,
            "howitzer_shells": 1,
            "howitzer_charges": 2,
            "high_explosive_ammunition": 1,
        },
    )

    actions = [
        ("Fire Missile completed", {}),
        ("Fire Missile completed", {}),  # depleted already
        ("Land Mine Deploy completed", {}),
        ("Decoy Deploy completed", {}),
        ("Smoke Deploy completed", {}),
        ("Howitzer Load Shell completed", {}),
        ("Howitzer Load Charges completed", {"charges": 3}),
        ("Autocannon Load completed", {"ammo_type": "high_explosive"}),
        ("Fire Cannon completed", {}),
        ("Fire Cannon completed", {}),  # depleted already
    ]
    for index, (subject, details) in enumerate(actions):
        sim.apply_message(completed(f"action-{index}", subject, details))

    state = sim.get_drone("drone-1")
    assert state is not None
    assert state.consumables == {
        "wire_guided_missile": 0,
        "land_mine": 0,
        "decoy_beacon": 0,
        "smoke_launcher": 0,
        "howitzer_shells": 0,
        "howitzer_charges": 0,
        "high_explosive_ammunition": 0,
    }
    assert all(count >= 0 for count in state.consumables.values())


def test_duplicate_completion_is_idempotent() -> None:
    sim = EntitySim()
    sim.seed_drone(
        "drone-1",
        current_battery=50,
        max_battery=50,
        consumables={"wire_guided_missile": 2},
    )
    message = completed("missile-1", "Fire Missile completed", {"level": 1})

    first = sim.apply_message(message)
    second = sim.apply_message(message)

    state = sim.get_drone("drone-1")
    assert first is not None and first.battery_spent == 10
    assert second is not None and second.duplicate
    assert state is not None
    assert state.current_battery == 40
    assert state.consumables["wire_guided_missile"] == 1


def test_building_costs_debit_command_center_stock_and_clamp() -> None:
    sim = EntitySim()
    sim.seed_building(
        "cc",
        building_type="command_center",
        stored_resources={
            "titanium_parts": 150,
            "battery_materials": 60,
            "enriched_unobtanium": 0,
        },
    )
    sim.seed_building("plant", building_type="drone_equipment_production_plant")

    build = sim.apply_message(
        building_completed(
            "build-1",
            "cc",
            "Building Build completed",
            {"building_type": "laser_turret"},
        )
    )
    manufacture = sim.apply_message(
        building_completed(
            "make-1",
            "plant",
            "Manufacture completed",
            {"equipment_type": "laser_cannon"},
        )
    )

    state = sim.get_building("cc")
    assert build is not None and build.resources_spent == {
        "titanium_parts": 100,
        "battery_materials": 40,
    }
    # Only 50 titanium and 20 battery remain, so the second debit clamps there.
    assert manufacture is not None and manufacture.resources_spent == {
        "titanium_parts": 18,
        "battery_materials": 12,
    }
    assert state is not None
    assert state.stored_resources == {
        "titanium_parts": 32,
        "battery_materials": 8,
        "enriched_unobtanium": 0,
    }
    assert all(value >= 0 for value in state.stored_resources.values())


def test_building_status_and_status_report_replace_local_stock() -> None:
    sim = EntitySim()
    sim.seed_building("cc", building_type="command_center", stored_resources={"titanium_parts": 1})

    sim.apply_message(
        building_completed(
            "status-1",
            "cc",
            "Building status completed",
            {
                "status": {
                    "building_id": "cc",
                    "building_type": "command_center",
                    "stored_resources": {
                        "titanium_parts": 90,
                        "battery_materials": 40,
                        "enriched_unobtanium": 2,
                    },
                }
            },
        )
    )
    report = sim.apply_message(
        building_completed(
            "report-1",
            "cc",
            "Building status_report completed",
            {
                "summary": {
                    "command_center_resources": {
                        "titanium_parts": 80,
                        "battery_materials": 35,
                        "enriched_unobtanium": 2,
                    }
                }
            },
        )
    )

    state = sim.get_building("cc")
    assert report is not None and report.status_checkpoint
    assert state is not None
    assert state.stored_resources["titanium_parts"] == 80


def test_unknown_typed_completion_preserves_raw_payload_for_simulation() -> None:
    sim = EntitySim()
    sim.seed_drone("drone-1", current_battery=100, max_battery=100)
    raw = completed("mine-1", "Land Mine Deploy completed", {"level": 2})
    typed = parse_wire_message(raw)

    delta = sim.apply_message(typed)

    assert delta is not None
    assert delta.battery_spent == 45


async def test_bus_attachment_consumes_wire_messages() -> None:
    bus = EventBus()
    sim = EntitySim(bus)
    sim.seed_drone("drone-1", current_battery=10, max_battery=10)

    await bus.publish(TOPIC_MESSAGE, completed("turn-1", "Turn completed", {"level": 1}))

    assert sim.get_drone("drone-1").current_battery == 8  # type: ignore[union-attr]
    sim.close()
