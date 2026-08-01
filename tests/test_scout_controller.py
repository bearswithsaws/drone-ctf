from __future__ import annotations

from agent.planning.controllers.scout import ScoutConfig, ScoutController, ScoutPhase
from agent.planning.tasks import ScoutSector
from agent.rules.hexmath import hex_distance_cube
from agent.sim.entity_sim import EntitySim
from agent.world.model import TileObservation, WorldModel
from agent.world.tiles import Terrain
from agent.world.tracks import Sighting, SightingSource, TrackStore


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


CONTACT = (6, -3)


async def _open_field(
    *, battery: int = 1000, threat_cost=None, config: ScoutConfig | None = None
) -> tuple[WorldModel, TrackStore, ScoutController]:
    """A scout in the middle of a fully-known clearing, so a retreat plan is
    limited by policy rather than by missing terrain."""

    world = WorldModel()
    sim = EntitySim()
    tracks = TrackStore()
    await world.upsert_drone("scout", q=0, r=0, direction=0, cycle=5)
    await world.observe_tiles(
        [
            TileObservation(q=q, r=r, terrain_type=Terrain.NORMAL)
            for q in range(-4, 5)
            for r in range(-4, 5)
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
        tracks=tracks,
        threat_cost=threat_cost,
        config=config,
    )
    return world, tracks, controller


async def test_scout_retreats_from_a_hostile_contact() -> None:
    _world, tracks, controller = await _open_field()
    await tracks.observe(Sighting(SightingSource.SCAN, q=CONTACT[0], r=CONTACT[1], cycle=5))

    plan = controller.plan()

    assert plan.phase is ScoutPhase.RETREATING
    assert plan.destination is not None
    # The leg opens the range and never spends battery scanning under contact.
    assert hex_distance_cube(*plan.destination, *CONTACT) > hex_distance_cube(0, 0, *CONTACT)
    assert all(action.action != "sensors/scan" for action in plan.actions)
    assert [contact.coord for contact in plan.contacts] == [CONTACT]
    assert "hostile contact" in plan.reason


async def test_scout_retreat_leg_respects_its_tile_budget() -> None:
    _world, tracks, controller = await _open_field(config=ScoutConfig(max_retreat_tiles=2))
    await tracks.observe(Sighting(SightingSource.SCAN, q=CONTACT[0], r=CONTACT[1], cycle=5))

    plan = controller.plan()

    assert plan.phase is ScoutPhase.RETREATING
    drives = [action for action in plan.actions if action.action == "propulsion/drive"]
    assert 0 < len(drives) <= 2


async def test_scout_retreats_out_of_a_weapon_envelope_without_a_track() -> None:
    # An enemy turret revealed by the last scan covers everything from q >= 0.
    _world, _tracks, controller = await _open_field(
        threat_cost=lambda coord: 1.0 if coord[0] >= 0 else 0.0
    )

    plan = controller.plan()

    assert plan.phase is ScoutPhase.RETREATING
    assert plan.destination is not None and plan.destination[0] < 0
    assert not plan.contacts
    assert "weapon envelope" in plan.reason


async def test_scout_keeps_exploring_past_a_decoy_contact() -> None:
    _world, tracks, controller = await _open_field()
    await tracks.observe(
        Sighting(SightingSource.IDENTIFY, q=1, r=0, cycle=5, drone_id="e1", is_decoy=True)
    )

    plan = controller.plan()

    assert plan.phase is ScoutPhase.TRAVELLING
    assert not plan.contacts
    assert plan.actions[-1].action == "sensors/scan"


async def test_scout_ignores_contacts_beyond_the_contact_radius() -> None:
    _world, tracks, controller = await _open_field(config=ScoutConfig(contact_radius=3))
    await tracks.observe(Sighting(SightingSource.SCAN, q=CONTACT[0], r=CONTACT[1], cycle=5))

    assert not controller.contacts()
    assert controller.plan().phase is ScoutPhase.TRAVELLING


async def test_scout_reports_blocked_when_no_retreat_tile_is_known() -> None:
    world = WorldModel()
    sim = EntitySim()
    tracks = TrackStore()
    await world.upsert_drone("scout", q=0, r=0, direction=0, cycle=5)
    await world.observe_tiles([TileObservation(q=0, r=0, terrain_type=Terrain.NORMAL)], cycle=5)
    sim.seed_drone(
        "scout",
        current_battery=1000,
        max_battery=1000,
        equipment=("propulsion", "sensors"),
    )
    await tracks.observe(Sighting(SightingSource.SCAN, q=2, r=-1, cycle=5))

    plan = ScoutController(
        world, sim, "scout", ScoutSector("sector", (0, 0), radius=8), tracks=tracks
    ).plan()

    assert plan.phase is ScoutPhase.BLOCKED
    assert not plan.actions
    assert plan.contacts and "no safer tile" in plan.reason


async def test_scout_retreat_may_spend_the_scouting_battery_reserve() -> None:
    # 100 battery blocks a normal frontier leg (see the reserve test above);
    # breaking contact still has to be planned.
    _world, tracks, controller = await _open_field(battery=100)
    await tracks.observe(Sighting(SightingSource.SCAN, q=CONTACT[0], r=CONTACT[1], cycle=5))

    plan = controller.plan()

    assert plan.phase is ScoutPhase.RETREATING
    assert plan.actions


async def test_scout_waits_without_required_equipment_checkpoint() -> None:
    world = WorldModel()
    sim = EntitySim()
    await world.upsert_drone("scout", q=0, r=0, direction=0, cycle=1)

    plan = ScoutController(world, sim, "scout", ScoutSector("sector", (1, 0))).plan()

    assert plan.phase is ScoutPhase.WAITING
    assert not plan.actions
