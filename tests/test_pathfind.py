"""Cost-optimal A*, threat avoidance, and action compilation tests."""

from __future__ import annotations

import pytest

from agent.planning.pathfind import (
    MovementProfile,
    PathfindingConfig,
    Pathfinder,
    compile_path,
)
from agent.rules.hexmath import hex_distance_cube
from agent.world import Sighting, SightingSource, ThreatMap, TrackStore, WorldModel
from agent.world.model import TileObservation
from agent.world.tiles import Terrain


DIRECT = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4))
DETOUR = ((0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (0, 4))
CORRIDORS = set(DIRECT) | set(DETOUR)


async def _observe(
    world: WorldModel,
    coords: set[tuple[int, int]],
    *,
    difficult: set[tuple[int, int]] | None = None,
) -> None:
    difficult = difficult or set()
    await world.observe_tiles(
        [
            TileObservation(
                q=q,
                r=r,
                terrain_type=(
                    Terrain.DIFFICULT if (q, r) in difficult else Terrain.NORMAL
                ),
            )
            for q, r in coords
        ],
        cycle=1,
    )


async def test_astar_chooses_optimal_cost_weighted_terrain_path() -> None:
    world = WorldModel()
    await _observe(world, CORRIDORS, difficult={(0, 1), (0, 2), (0, 3)})
    finder = Pathfinder(
        world,
        config=PathfindingConfig(
            threat_weight=0,
            comms_weight=0,
            unknown_penalty=0,
        ),
    )

    result = finder.find_path(DIRECT[0], DIRECT[-1], allowed=CORRIDORS)

    assert result is not None
    assert result.path == DETOUR
    assert result.cost == 5


async def test_unknown_and_comms_costs_are_part_of_path_selection() -> None:
    world = WorldModel()
    # The direct corridor is unexplored; the longer detour is known safe terrain.
    await _observe(world, set(DETOUR))
    risky = {(0, 1), (0, 2), (0, 3)}
    finder = Pathfinder(
        world,
        comms_risk=lambda coord: 1.0 if coord in risky else 0.0,
        config=PathfindingConfig(
            threat_weight=0,
            comms_weight=2,
            unknown_penalty=2,
        ),
    )

    result = finder.find_path(DIRECT[0], DIRECT[-1], allowed=CORRIDORS)

    assert result is not None
    assert result.path == DETOUR
    assert result.cost == 5


async def test_known_buildings_block_ground_paths_but_overflight_can_be_enabled() -> None:
    world = WorldModel()
    await _observe(world, CORRIDORS)
    await world.observe_tile(
        TileObservation(
            q=0,
            r=2,
            terrain_type=Terrain.NORMAL,
            has_building=True,
            building_id="obstacle",
        ),
        cycle=2,
    )

    ground = Pathfinder(world).find_path(DIRECT[0], DIRECT[-1], allowed=CORRIDORS)
    airborne = Pathfinder(
        world,
        movement=MovementProfile(
            ignores_difficult_terrain=True,
            can_overfly_buildings=True,
        ),
    ).find_path(DIRECT[0], DIRECT[-1], allowed=CORRIDORS)

    assert ground is not None
    assert (0, 2) not in ground.path
    assert len(ground.path) > len(DIRECT)
    assert airborne is not None
    assert airborne.path == DIRECT


async def test_pathfinder_uses_world_snapshot_captured_at_construction() -> None:
    world = WorldModel()
    await _observe(world, set(DIRECT))
    finder = Pathfinder(world)

    await world.observe_tile(
        TileObservation(q=0, r=4, terrain_type=Terrain.IMPASSABLE),
        cycle=2,
    )

    captured = finder.find_path(DIRECT[0], DIRECT[-1], allowed=set(DIRECT))
    current = Pathfinder(world).find_path(DIRECT[0], DIRECT[-1], allowed=set(DIRECT))
    assert captured is not None
    assert captured.path == DIRECT
    assert current is None


async def test_pathfinder_snapshots_cost_providers_that_support_it() -> None:
    world = WorldModel()
    await _observe(world, set(DIRECT))

    class MutableCost:
        value = 0.0

        def __call__(self, _coord: tuple[int, int]) -> float:
            return self.value

        def snapshot(self):
            captured = self.value
            return lambda _coord: captured

    risk = MutableCost()
    finder = Pathfinder(
        world,
        comms_risk=risk,
        config=PathfindingConfig(comms_weight=10, unknown_penalty=0),
    )
    risk.value = 100.0

    result = finder.find_path(DIRECT[0], DIRECT[-1], allowed=set(DIRECT))
    assert result is not None
    assert result.cost == 4


async def test_airborne_profile_ignores_difficult_terrain_cost() -> None:
    world = WorldModel()
    await _observe(world, set(DIRECT), difficult={(0, 1), (0, 2), (0, 3)})

    ground = Pathfinder(world).find_path(DIRECT[0], DIRECT[-1], allowed=set(DIRECT))
    airborne = Pathfinder(
        world,
        movement=MovementProfile(ignores_difficult_terrain=True),
    ).find_path(DIRECT[0], DIRECT[-1], allowed=set(DIRECT))

    assert ground is not None
    assert ground.cost == 7
    assert airborne is not None
    assert airborne.cost == 4


@pytest.mark.parametrize("source_kind", ["turret", "howitzer"])
async def test_path_avoids_known_weapon_envelope_when_safe_route_exists(
    source_kind: str,
) -> None:
    world = WorldModel()
    tracks = TrackStore()
    if source_kind == "turret":
        await world.upsert_enemy_building(
            "enemy-turret",
            building_type="laser_turret",
            origin=(0, 0),
            cycle=1,
        )
        start, goal = (-16, -8), (16, 8)
    else:
        await tracks.observe(
            Sighting(
                SightingSource.IDENTIFY,
                0,
                0,
                1,
                drone_id="enemy-artillery",
                equipment=frozenset({"howitzer"}),
            )
        )
        start, goal = (-33, -17), (33, 16)

    threat = ThreatMap(world, tracks)
    finder = Pathfinder(
        world,
        threat_cost=threat,
        config=PathfindingConfig(comms_weight=0, unknown_penalty=0),
    )

    result = finder.find_path(start, goal)

    assert result is not None
    assert all(threat(coord) == 0 for coord in result.path)
    assert len(result.path) > hex_distance_cube(*start, *goal) + 1


def test_compile_path_emits_shortest_turns_and_forward_drives() -> None:
    path = ((0, 0), (1, 0), (2, 0), (2, 1))

    actions = compile_path(path, initial_heading=5, level=2)

    assert [(action.action, dict(action.payload)) for action in actions] == [
        ("propulsion/turn", {"direction": 1, "level": 2}),
        ("propulsion/turn", {"direction": 1, "level": 2}),
        ("propulsion/drive", {"direction": 1, "level": 2}),
        ("propulsion/turn", {"direction": -1, "level": 2}),
        ("propulsion/drive", {"direction": 1, "level": 2}),
        ("propulsion/turn", {"direction": 1, "level": 2}),
        ("propulsion/turn", {"direction": 1, "level": 2}),
        ("propulsion/drive", {"direction": 1, "level": 2}),
    ]


async def test_compiled_motion_retries_fail_closed_without_state_guards() -> None:
    actions = compile_path(((0, 0), (1, 0)), initial_heading=1)

    assert len(actions) == 1
    assert await actions[0].precondition() is False


def test_compile_path_binds_expected_state_retry_guards() -> None:
    expected_states: list[tuple[tuple[int, int], int]] = []

    def guard(position: tuple[int, int], heading: int):
        expected_states.append((position, heading))
        return lambda: True

    actions = compile_path(
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        initial_heading=5,
        precondition_factory=guard,
    )

    assert expected_states == [
        ((0, 0), 5),
        ((0, 0), 0),
        ((0, 0), 1),
        ((1, 0), 1),
        ((1, 0), 0),
        ((2, 0), 0),
        ((2, 0), 1),
        ((2, 0), 2),
    ]
    assert all(action.precondition() for action in actions)


def test_compile_path_rejects_non_adjacent_steps() -> None:
    with pytest.raises(ValueError, match="non-adjacent"):
        compile_path(((0, 0), (2, 0)), initial_heading=0)


@pytest.mark.parametrize(("heading", "level"), [(1.5, 1), (0, 1.5), (True, 1)])
def test_compile_path_rejects_non_integer_controls(heading: object, level: object) -> None:
    with pytest.raises(ValueError):
        compile_path(((0, 0),), heading, level=level)  # type: ignore[arg-type]


async def test_impassable_goal_has_no_path() -> None:
    world = WorldModel()
    await world.observe_tile(
        TileObservation(q=1, r=0, terrain_type=Terrain.IMPASSABLE),
        cycle=1,
    )

    assert Pathfinder(world).find_path((0, 0), (1, 0)) is None
