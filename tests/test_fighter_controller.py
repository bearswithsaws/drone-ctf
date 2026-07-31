"""Engagement, range-control, and survival tests for fighter micro."""

from __future__ import annotations

from agent.planning.controllers.fighter import FighterController, FighterPhase
from agent.rules.hexmath import hex_distance_cube
from agent.sim.entity_sim import EntitySim
from agent.world.model import TileObservation, WorldModel
from agent.world.tiles import Terrain
from agent.world.tracks import Sighting, SightingSource, TrackStore


async def make_fighter(
    *,
    fighter_coord: tuple[int, int] = (0, 0),
    fighter_heading: int = 3,
    target_coord: tuple[int, int] = (4, 0),
    current_health: int = 100,
    current_battery: int = 1000,
    equipment: tuple[str, ...] = ("propulsion", "laser_cannon", "targeting_computer"),
    consumables: dict[str, int] | None = None,
    loaded_ammo_type: str | None = None,
    marked_targets: set[str] | None = None,
):
    world = WorldModel()
    tracks = TrackStore()
    sim = EntitySim()
    await world.observe_tiles(
        [
            TileObservation(q=q, r=r, terrain_type=Terrain.NORMAL, elevation=0)
            for q in range(-12, 13)
            for r in range(-8, 9)
        ],
        cycle=1,
    )
    await world.upsert_drone(
        "fighter-1",
        q=fighter_coord[0],
        r=fighter_coord[1],
        direction=fighter_heading,
        elevation=0,
        cycle=1,
    )
    sim.seed_drone(
        "fighter-1",
        current_battery=current_battery,
        max_battery=1000,
        current_health=current_health,
        max_health=100,
        total_weight=80,
        equipment=equipment,
        functional_equipment=equipment,
        equipment_levels={item: 1 for item in equipment},
        consumables=consumables,
        loaded_ammo_type=loaded_ammo_type,
    )
    target = await tracks.observe(
        Sighting(
            SightingSource.IDENTIFY,
            target_coord[0],
            target_coord[1],
            1,
            drone_id="enemy-1",
            equipment=frozenset({"laser_cannon"}),
        )
    )
    controller = FighterController(
        world,
        tracks,
        sim,
        "fighter-1",
        marked_targets=marked_targets or set(),
    )
    return world, tracks, sim, target, controller


async def test_turn_mark_and_laser_fire_are_pipelined_at_preferred_range() -> None:
    _world, _tracks, _sim, _target, controller = await make_fighter()

    plan = controller.plan()

    assert plan.phase is FighterPhase.ENGAGING
    assert [action.action for action in plan.actions] == [
        "propulsion/turn",
        "propulsion/turn",
        "propulsion/turn",
        "targeting_computer/mark",
        "laser_cannon/fire",
    ]
    assert dict(plan.actions[-1].payload) == {
        "target_q": 4,
        "target_r": 0,
        "level": 1,
    }
    assert plan.preferred_min_range == 3
    assert plan.preferred_max_range == 5
    assert plan.actions[0].precondition()


async def test_auto_cannon_loads_then_fires_loaded_ammunition() -> None:
    world, tracks, sim, target, controller = await make_fighter(
        equipment=("propulsion", "auto_cannon"),
        consumables={"armour_piercing_ammunition": 3},
    )

    arming = controller.plan()
    assert arming.phase is FighterPhase.ARMING
    assert [action.action for action in arming.actions] == ["auto_cannon/load"]
    assert dict(arming.actions[0].payload) == {"ammo_type": "armour_piercing"}

    sim.seed_drone(
        "fighter-1",
        current_battery=1000,
        max_battery=1000,
        current_health=100,
        max_health=100,
        total_weight=80,
        equipment=("propulsion", "auto_cannon"),
        functional_equipment=("propulsion", "auto_cannon"),
        equipment_levels={"auto_cannon": 1},
        consumables={"armour_piercing_ammunition": 3},
        loaded_ammo_type="armour_piercing_ammunition",
    )
    loaded = FighterController(world, tracks, sim, "fighter-1").plan()

    assert loaded.target is not None and loaded.target.track_id == target.track_id
    assert loaded.phase is FighterPhase.ENGAGING
    assert "auto_cannon/fire" in [action.action for action in loaded.actions]


async def test_range_control_closes_and_opens_to_the_preferred_band() -> None:
    _world, _tracks, _sim, _target, closing_controller = await make_fighter(target_coord=(9, 4))

    closing = closing_controller.plan()
    assert closing.phase is FighterPhase.CLOSING
    assert closing.actions
    assert closing.destination is not None and closing.target is not None
    assert 3 <= hex_distance_cube(*closing.destination, *closing.target.coord) <= 5

    _world, _tracks, _sim, _target, opening_controller = await make_fighter(target_coord=(1, 0))
    opening = opening_controller.plan()
    assert opening.phase is FighterPhase.OPENING
    assert opening.actions
    assert opening.destination is not None and opening.target is not None
    assert 3 <= hex_distance_cube(*opening.destination, *opening.target.coord) <= 5


async def test_blocked_shot_repositions_to_a_clear_firing_line() -> None:
    world, _tracks, _sim, _target, controller = await make_fighter()
    await world.observe_tile(
        TileObservation(
            q=2,
            r=0,
            terrain_type=Terrain.IMPASSABLE,
            elevation=0,
        ),
        cycle=2,
    )

    plan = controller.plan()

    assert plan.phase is FighterPhase.REPOSITIONING
    assert plan.actions
    assert plan.destination != (0, 0)


async def test_health_and_battery_thresholds_disengage_without_firing() -> None:
    for health, battery in ((30, 1000), (100, 200)):
        world, _tracks, _sim, _target, controller = await make_fighter(
            current_health=health,
            current_battery=battery,
        )
        await world.upsert_building(
            "cc",
            building_type="command_center",
            origin=(-5, -2),
            tiles=((-5, -2),),
            cycle=2,
        )
        await world.observe_tile(
            TileObservation(
                q=-5,
                r=-2,
                terrain_type=Terrain.NORMAL,
                elevation=0,
                has_building=True,
                building_id="cc",
            ),
            cycle=2,
        )

        plan = controller.plan()

        assert plan.phase is FighterPhase.DISENGAGING
        assert plan.actions
        assert all(not action.action.endswith("/fire") for action in plan.actions)
        assert "threshold crossed" in plan.reason


async def test_marked_target_is_focused_over_a_nearer_unmarked_contact() -> None:
    world, tracks, sim, _target, _controller = await make_fighter(target_coord=(2, 0))
    marked = await tracks.observe(
        Sighting(
            SightingSource.IDENTIFY,
            4,
            0,
            1,
            drone_id="enemy-marked",
            equipment=frozenset({"laser_cannon"}),
        )
    )

    plan = FighterController(
        world,
        tracks,
        sim,
        "fighter-1",
        marked_targets={marked.track_id},
    ).plan()

    assert plan.target is not None
    assert plan.target.target_id == "enemy-marked"
    assert plan.target.marked


async def test_stale_or_decoy_tracks_are_not_engaged() -> None:
    world, tracks, sim, target, _controller = await make_fighter()
    target.is_decoy = True

    decoy = FighterController(world, tracks, sim, "fighter-1").plan()
    assert decoy.phase is FighterPhase.WAITING
    assert decoy.actions == ()

    target.is_decoy = False
    await world.set_cycle(500)
    stale = FighterController(world, tracks, sim, "fighter-1").plan()
    assert stale.phase is FighterPhase.WAITING
    assert stale.actions == ()
