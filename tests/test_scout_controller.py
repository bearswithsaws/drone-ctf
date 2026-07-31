from __future__ import annotations

from agent.planning.controllers.scout import ScoutController, ScoutPhase
from agent.planning.tasks import ScoutSector
from agent.sim.entity_sim import EntitySim
from agent.world.model import TileObservation, WorldModel
from agent.world.tiles import Terrain


async def test_scout_moves_to_stale_frontier_then_scans() -> None:
    world = WorldModel()
    sim = EntitySim()
    await world.upsert_drone("scout", q=0, r=0, direction=0, cycle=5)
    await world.observe_tile(
        TileObservation(q=0, r=0, terrain_type=Terrain.NORMAL), cycle=5
    )
    sim.seed_drone(
        "scout",
        current_battery=100,
        equipment=("propulsion", "sensors"),
        equipment_levels={"propulsion": 2, "sensors": 2},
    )
    controller = ScoutController(world, sim, "scout", ScoutSector("sector", (2, 0), radius=0))

    travelling = controller.plan()

    assert travelling.phase is ScoutPhase.TRAVELLING
    assert travelling.destination == (2, 0)
    assert travelling.actions
    assert travelling.actions[-1].action == "propulsion/drive"

    await world.upsert_drone("scout", q=2, r=0, direction=0, cycle=6)
    scanning = controller.plan()

    assert scanning.phase is ScoutPhase.SCANNING
    assert [action.action for action in scanning.actions] == ["sensors/scan"]
    assert dict(scanning.actions[0].payload) == {"level": 2}


async def test_scout_waits_without_required_equipment_checkpoint() -> None:
    world = WorldModel()
    sim = EntitySim()
    await world.upsert_drone("scout", q=0, r=0, direction=0, cycle=1)

    plan = ScoutController(world, sim, "scout", ScoutSector("sector", (1, 0))).plan()

    assert plan.phase is ScoutPhase.WAITING
    assert not plan.actions
