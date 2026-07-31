"""Cost-optimal A*, threat avoidance, and action compilation tests."""

from __future__ import annotations

import pytest

from agent.planning.pathfind import PathfindingConfig, Pathfinder, compile_path
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
