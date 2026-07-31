from __future__ import annotations

from agent.planning.controllers.scout import ScoutController, ScoutPhase
from agent.planning.tasks import ScoutSector
from agent.sim.entity_sim import EntitySim
from agent.world.model import TileObservation, WorldModel
from agent.world.tiles import Terrain


async def _scout(
    *, battery: int = 1000, threat_cost=None
) -> tuple[WorldModel, EntitySim, ScoutController]:
    world = WorldModel()
    sim = EntitySim()
    await world.upsert_drone("scout", q=0, r=0, direction=0, cycle=5)
    await world.observe_tiles(
        [
            TileObservation(q=0, r=0, terrain_type=Terrain.NORMAL),
            TileObservation(q=1, r=-1, terrain_type=Terrain.NORMAL),
            TileObservation(q=2, r=-1, terrain_type=Terrain.NORMAL),
        ],
        cycle=5,
    )
    sim.seed_drone(
        "scout",
        current_battery=battery,
        max_battery=1000,
        equipment=("propulsion", "sensors"),
        equipment_levels={"propulsion": 1, "sensors": 2},
    )
    controller = ScoutController(
        world,
        sim,
        "scout",
        ScoutSector("sector", (0, 0), radius=8),
        threat_cost=threat_cost,
    )
    return world, sim, controller


async def test_scout_moves_over_known_tiles_to_frontier_then_scans() -> None:
    _world, _sim, controller = await _scout()

    plan = controller.plan()

    assert plan.phase is ScoutPhase.TRAVELLING
    assert plan.destination == (2, -1)
    assert [action.action for action in plan.actions] == [
        "propulsion/drive",
        "propulsion/drive",
        "sensors/scan",
    ]
    assert dict(plan.actions[-1].payload) == {"level": 2}


async def test_replan_advances_to_newly_revealed_frontier() -> None:
    world, _sim, controller = await _scout()
    first = controller.plan()
    assert first.destination == (2, -1)

    await world.upsert_drone("scout", q=2, r=-1, direction=0, cycle=6)
    await world.observe_tiles(
        [
            TileObservation(q=2, r=-1, terrain_type=Terrain.NORMAL),
            TileObservation(q=3, r=-2, terrain_type=Terrain.NORMAL),
        ],
        cycle=6,
    )

    second = controller.plan()
    assert second.destination == (3, -2)
    assert [action.action for action in second.actions] == [
        "propulsion/drive",
        "sensors/scan",
    ]


async def test_scout_scans_in_place_before_any_unknown_movement() -> None:
    world, _sim, controller = await _scout()
    # Make every known destination except the occupied tile impassable.
    await world.observe_tiles(
        [
            TileObservation(q=1, r=-1, terrain_type=Terrain.IMPASSABLE),
            TileObservation(q=2, r=-1, terrain_type=Terrain.IMPASSABLE),
        ],
        cycle=6,
    )

    plan = controller.plan()
    assert plan.phase is ScoutPhase.SCANNING
    assert plan.destination == (0, 0)
    assert [action.action for action in plan.actions] == ["sensors/scan"]


async def test_scout_avoids_threatened_frontier() -> None:
    _world, _sim, controller = await _scout(
        threat_cost=lambda coord: 1.0 if coord == (2, -1) else 0.0
    )

    plan = controller.plan()

    assert plan.destination == (1, -1)
    assert all(action.action != "propulsion/turn" for action in plan.actions)


async def test_scout_blocks_before_violating_battery_reserve() -> None:
    _world, _sim, controller = await _scout(battery=100)

    plan = controller.plan()

    assert plan.phase is ScoutPhase.BLOCKED
    assert not plan.actions
    assert "battery reserve" in plan.reason


async def test_scout_waits_without_required_equipment_checkpoint() -> None:
    world = WorldModel()
    sim = EntitySim()
    await world.upsert_drone("scout", q=0, r=0, direction=0, cycle=1)

    plan = ScoutController(world, sim, "scout", ScoutSector("sector", (1, 0))).plan()

    assert plan.phase is ScoutPhase.WAITING
    assert not plan.actions
