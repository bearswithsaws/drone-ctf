from __future__ import annotations

from agent.planning.controllers.charger import ChargerController
from agent.planning.controllers.miner import MinerPhase
from agent.planning.tasks import Recharge
from agent.sim.entity_sim import EntitySim
from agent.transport.action_tracker import EntityKind
from agent.world.model import TileObservation, WorldModel
from agent.world.tiles import Terrain


async def _context(position=(0, 0), battery=200):
    world = WorldModel()
    sim = EntitySim()
    await world.upsert_building(
        "charger",
        building_type="drone_charging_station",
        origin=(-2, 0),
        tiles=((-2, 0),),
        cycle=1,
    )
    await world.upsert_drone("d1", q=position[0], r=position[1], direction=3, cycle=1)
    await world.observe_tiles(
        [
            TileObservation(q=0, r=0, terrain_type=Terrain.NORMAL),
            TileObservation(q=-1, r=0, terrain_type=Terrain.NORMAL),
        ],
        cycle=1,
    )
    sim.seed_drone(
        "d1",
        current_battery=battery,
        max_battery=1000,
        equipment=("propulsion",),
    )
    task = Recharge("recharge:d1", drone_id="d1", target_fraction=0.9)
    return world, sim, ChargerController(world, sim, "d1", task)


async def test_low_battery_drone_routes_to_known_charger_service_tile() -> None:
    _world, _sim, controller = await _context()

    plan = controller.plan()

    assert plan.phase is MinerPhase.SEEK_CHARGER
    assert plan.destination == (-1, 0)
    assert plan.entity_key == (EntityKind.DRONE, "d1")
    assert plan.actions_for(EntityKind.DRONE, "d1")[-1].action == "propulsion/drive"
    assert plan.required_battery > plan.reserve_battery


async def test_station_charge_uses_station_relative_coordinates_and_recovery_target() -> None:
    _world, _sim, controller = await _context(position=(-1, 0), battery=200)

    plan = controller.plan()

    assert plan.phase is MinerPhase.CHARGING
    assert plan.entity_key == (EntityKind.BUILDING, "charger")
    actions = plan.actions_for(EntityKind.BUILDING, "charger")
    assert len(actions) == 3
    assert dict(actions[0].payload) == {"q": 1, "r": 0, "efficiency": 1.0}
    assert plan.required_battery == 900
