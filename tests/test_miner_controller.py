"""State-machine and safety tests for the autonomous miner controller."""

from __future__ import annotations

from agent.planning.controllers.miner import (
    MinerController,
    MinerPhase,
    group_miner_actions,
)
from agent.planning.tasks import MineLoop
from agent.rules.hexmath import get_neighbor
from agent.sim.entity_sim import EntitySim
from agent.transport.action_tracker import EntityKind
from agent.world.model import TileObservation, WorldModel
from agent.world.tiles import Terrain


RESOURCE = (5, 1)
REFINER = (0, 0)
CHARGER = (-4, -1)


async def make_world(*, drone_position: tuple[int, int], battery: int = 4000):
    world = WorldModel()
    sim = EntitySim()
    known = {(q, r) for q in range(-6, 8) for r in range(-4, 5)}
    await world.observe_tiles(
        [TileObservation(q=q, r=r, terrain_type=Terrain.NORMAL) for q, r in known],
        cycle=1,
    )
    await world.observe_tile(
        TileObservation(
            q=RESOURCE[0],
            r=RESOURCE[1],
            terrain_type=Terrain.NORMAL,
            has_resource=True,
            resource_type="titanium_ore",
            resource_amount=500,
        ),
        cycle=2,
    )
    for building_id, building_type, coord in (
        ("refiner", "refiner", REFINER),
        ("charger", "drone_charging_station", CHARGER),
    ):
        await world.upsert_building(
            building_id,
            building_type=building_type,
            origin=coord,
            tiles=(coord,),
            cycle=2,
        )
        await world.observe_tile(
            TileObservation(
                q=coord[0],
                r=coord[1],
                terrain_type=Terrain.NORMAL,
                has_building=True,
                building_id=building_id,
            ),
            cycle=2,
        )
        sim.seed_building(building_id, building_type=building_type)

    await world.upsert_drone(
        "miner-1",
        q=drone_position[0],
        r=drone_position[1],
        direction=0,
        cycle=2,
    )
    sim.seed_drone(
        "miner-1",
        current_battery=battery,
        max_battery=4000,
        total_weight=100,
        hopper_capacity=25,
    )
    task = MineLoop(
        "mine-titanium",
        resource_node=RESOURCE,
        resource_type="titanium_ore",
        dropoff_id="refiner",
    )
    return world, sim, MinerController(world, sim, "miner-1", task)


async def move_drone(world: WorldModel, position: tuple[int, int]) -> None:
    await world.upsert_drone(
        "miner-1",
        q=position[0],
        r=position[1],
        direction=0,
        cycle=world.cycle + 1,
    )


async def test_miner_plans_each_queue_safe_leg_of_the_full_loop() -> None:
    resource_service = get_neighbor(*RESOURCE, 0)
    world, sim, controller = await make_world(drone_position=resource_service)

    mine = controller.plan()
    assert mine.phase is MinerPhase.MINING
    assert mine.entity_key == (EntityKind.DRONE, "miner-1")
    assert mine.actions[0].action.action == "drill/mine"
    assert dict(mine.actions[0].action.payload) == {
        "level": 1,
        "resource_q": RESOURCE[0],
        "resource_r": RESOURCE[1],
    }
    assert mine.actions[0].action.precondition()

    sim.apply_message(
        {
            "message_type": "action_completed",
            "action_id": "mine-1",
            "drone_id": "miner-1",
            "subject": "Mine completed",
            "details": {
                "success": True,
                "level": 1,
                "ore_type": "titanium_ore",
                "total_ore_mined": 25,
            },
        }
    )
    returning = controller.plan()
    assert returning.phase is MinerPhase.RETURNING
    assert returning.entity_key == (EntityKind.DRONE, "miner-1")
    assert returning.actions_for(EntityKind.DRONE, "miner-1")
    assert all(action.entity_kind is EntityKind.DRONE for action in returning.actions)

    refiner_service = get_neighbor(*REFINER, 0)
    await move_drone(world, refiner_service)
    unload = controller.plan()
    assert unload.phase is MinerPhase.UNLOADING
    assert unload.entity_key == (EntityKind.BUILDING, "refiner")
    assert unload.actions[0].action.action == "common/unload"
    assert unload.actions_for(EntityKind.BUILDING, "refiner") == (unload.actions[0].action,)

    sim.apply_message(
        {
            "message_type": "building_action_completed",
            "action_id": "unload-1",
            "building_id": "refiner",
            "drone_id": "miner-1",
            "subject": "Unload completed",
            "details": {
                "success": True,
                "ore_unloaded": {"titanium_ore": 25},
            },
        }
    )
    sim.seed_drone(
        "miner-1",
        current_battery=100,
        max_battery=4000,
        total_weight=100,
        hopper_capacity=25,
    )
    charger_service = get_neighbor(*CHARGER, 0)
    await move_drone(world, charger_service)
    charging = controller.plan()
    assert charging.phase is MinerPhase.CHARGING
    assert charging.entity_key == (EntityKind.BUILDING, "charger")
    assert len(charging.actions) > 1
    assert all(action.action.action == "charging_station/charge" for action in charging.actions)
    assert charging.reserve_battery == 600


async def test_low_battery_routes_to_charger_before_starting_mining_trip() -> None:
    world, _sim, controller = await make_world(drone_position=(0, 2), battery=700)

    plan = controller.plan()

    assert plan.phase is MinerPhase.SEEK_CHARGER
    assert plan.entity_key == (EntityKind.DRONE, "miner-1")
    assert plan.required_battery > 700
    assert plan.reserve_battery == 600
    assert plan.actions
    assert plan.actions[0].action.precondition()


async def test_resource_selection_falls_back_to_nearer_matching_node() -> None:
    world, _sim, controller = await make_world(drone_position=(2, 1))
    nearer = (3, 1)
    await world.observe_tile(
        TileObservation(
            q=nearer[0],
            r=nearer[1],
            terrain_type=Terrain.NORMAL,
            has_resource=True,
            resource_type="titanium_ore",
            resource_amount=100,
        ),
        cycle=3,
    )

    plan = controller.plan()

    assert plan.resource_node == nearer
    assert plan.phase is MinerPhase.MINING


async def test_missing_checkpoint_waits_without_issuing_actions() -> None:
    world = WorldModel()
    sim = EntitySim()
    task = MineLoop("mine", resource_node=(1, 0), dropoff_id="refiner")

    plan = MinerController(world, sim, "miner-1", task).plan()

    assert plan.phase is MinerPhase.WAITING
    assert plan.actions == ()
    assert "position" in plan.reason


async def test_shared_building_actions_are_serialized_for_two_miners() -> None:
    refiner_service = get_neighbor(*REFINER, 0)
    world, sim, first = await make_world(drone_position=refiner_service)
    await world.upsert_drone(
        "miner-2",
        q=refiner_service[0],
        r=refiner_service[1],
        direction=0,
        cycle=3,
    )
    sim.seed_drone(
        "miner-1",
        current_battery=4000,
        max_battery=4000,
        total_weight=100,
        cargo={"titanium_ore": 25},
        hopper_capacity=25,
    )
    sim.seed_drone(
        "miner-2",
        current_battery=4000,
        max_battery=4000,
        total_weight=100,
        cargo={"titanium_ore": 25},
        hopper_capacity=25,
    )
    second = MinerController(world, sim, "miner-2", first.task)

    grouped = group_miner_actions((first.plan(), second.plan()))

    shared = grouped[(EntityKind.BUILDING, "refiner")]
    assert [action.action for action in shared] == ["common/unload", "common/unload"]
