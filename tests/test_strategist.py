"""Strategist tests: doctrine FSM, assessment, task generation, tick loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.planning.allocator import TaskAllocator
from agent.planning.strategist import (
    Doctrine,
    Posture,
    ScoreState,
    StrategicAssessment,
    Strategist,
    StrategistConfig,
    assess,
    default_fitness,
    generate_tasks,
    next_doctrine,
)
from agent.planning.tasks import MineLoop, ProduceDrone, Research, Strike
from agent.world import Sighting, SightingSource, ThreatMap, TileObservation, TrackStore, WorldModel


# --- test doubles ---

class FakePipeliner:
    def __init__(self) -> None:
        self.registered: dict[tuple[str, str], Any] = {}
        self.replaced: list[tuple[str, str]] = []

    def register(self, kind: str, entity_id: str, output: Any) -> None:
        key = (kind, entity_id)
        if key in self.registered:
            raise ValueError("already registered")
        self.registered[key] = output

    async def replace_plan(self, kind: str, entity_id: str, output: Any) -> None:
        self.replaced.append((kind, entity_id))
        self.registered[(kind, entity_id)] = output


@dataclass
class Ctx:
    world: WorldModel
    tracks: TrackStore
    threat_map: ThreatMap
    allocator: TaskAllocator
    pipeliner: FakePipeliner
    entity_sim: Any = None


async def _base_context() -> Ctx:
    world = WorldModel()
    tracks = TrackStore()
    threat = ThreatMap(world, tracks)
    # A command center (production-capable) with a footprint tile at (0,0).
    await world.upsert_building("cc", building_type="command_center", origin=(0, 0),
                                tiles=[(0, 0)], cycle=1)
    return Ctx(world, tracks, threat, TaskAllocator(), FakePipeliner())


def _assessment(**kw) -> StrategicAssessment:
    defaults = dict(
        own_drones=2, own_buildings=1, enemy_tracks=0, resource_nodes=1,
        under_attack=False, economy_ready=True, posture=Posture.NEUTRAL, score_lead=0,
    )
    defaults.update(kw)
    return StrategicAssessment(**defaults)


# --- doctrine FSM ---

def test_threat_forces_defend_from_any_posture() -> None:
    for posture in Posture:
        a = _assessment(under_attack=True, posture=posture)
        assert next_doctrine(Doctrine.EXPAND, a) is Doctrine.DEFEND


def test_behind_goes_all_in_when_economy_ready() -> None:
    a = _assessment(posture=Posture.AGGRESSIVE, economy_ready=True)
    assert next_doctrine(Doctrine.BUILD_UP, a) is Doctrine.ALL_IN


def test_behind_but_no_economy_builds_up() -> None:
    a = _assessment(posture=Posture.AGGRESSIVE, economy_ready=False)
    assert next_doctrine(Doctrine.BUILD_UP, a) is Doctrine.BUILD_UP


def test_protect_lead_expands_never_raids() -> None:
    a = _assessment(posture=Posture.PROTECT, economy_ready=True, enemy_tracks=3)
    assert next_doctrine(Doctrine.RAID, a) is Doctrine.EXPAND


def test_neutral_raids_when_ready_and_enemies_known() -> None:
    a = _assessment(posture=Posture.NEUTRAL, economy_ready=True, enemy_tracks=1)
    assert next_doctrine(Doctrine.BUILD_UP, a) is Doctrine.RAID


def test_neutral_expands_when_ready_and_no_enemies() -> None:
    a = _assessment(posture=Posture.NEUTRAL, economy_ready=True, enemy_tracks=0)
    assert next_doctrine(Doctrine.BUILD_UP, a) is Doctrine.EXPAND


def test_not_ready_builds_up() -> None:
    a = _assessment(posture=Posture.NEUTRAL, economy_ready=False)
    assert next_doctrine(Doctrine.EXPAND, a) is Doctrine.BUILD_UP


# --- assessment ---

async def test_assess_posture_from_score() -> None:
    ctx = await _base_context()
    cfg = StrategistConfig(protect_margin=50, aggro_margin=50)
    await ctx.world.upsert_drone("d1", q=0, r=0, cycle=1)
    await ctx.world.upsert_drone("d2", q=1, r=0, cycle=1)

    ahead = assess(ctx, ScoreState(own=100, best_enemy=40), cfg)
    assert ahead.posture is Posture.PROTECT
    assert ahead.economy_ready is True

    behind = assess(ctx, ScoreState(own=10, best_enemy=100), cfg)
    assert behind.posture is Posture.AGGRESSIVE

    even = assess(ctx, ScoreState(own=100, best_enemy=100), cfg)
    assert even.posture is Posture.NEUTRAL


async def test_assess_under_attack_when_enemy_near_base() -> None:
    ctx = await _base_context()
    cfg = StrategistConfig(base_defense_radius=8)
    # Enemy track 3 tiles from the base tile (0,0) -> under attack.
    await ctx.tracks.observe(Sighting(SightingSource.SCAN, 3, 0, 5))
    a = assess(ctx, ScoreState(), cfg)
    assert a.under_attack is True


async def test_assess_not_under_attack_when_enemy_far() -> None:
    ctx = await _base_context()
    cfg = StrategistConfig(base_defense_radius=8)
    await ctx.tracks.observe(Sighting(SightingSource.SCAN, 40, 40, 5))
    a = assess(ctx, ScoreState(), cfg)
    assert a.under_attack is False


async def test_assess_economy_not_ready_without_drones() -> None:
    ctx = await _base_context()
    cfg = StrategistConfig(min_economy_drones=2)
    a = assess(ctx, ScoreState(), cfg)  # no drones added
    assert a.economy_ready is False


# --- task generation ---

async def test_build_up_generates_economy_tasks() -> None:
    ctx = await _base_context()
    await ctx.world.observe_tile(
        TileObservation(q=5, r=5, has_resource=True, resource_type="titanium_ore"), cycle=1
    )
    a = _assessment(own_drones=1, resource_nodes=1)
    tasks = generate_tasks(Doctrine.BUILD_UP, ctx, a, StrategistConfig())
    kinds = {t.kind for t in tasks}
    assert "mine_loop" in kinds
    assert any(isinstance(t, Research) for t in tasks)
    assert any(isinstance(t, ProduceDrone) for t in tasks)


async def test_task_generation_advances_past_completed_research() -> None:
    ctx = await _base_context()
    ctx.entity_sim = type(
        "Sim",
        (),
        {"research_tiers": {"unlock_auto_cannon": 1, "baseline_drone": 3}},
    )()

    tasks = generate_tasks(Doctrine.BUILD_UP, ctx, _assessment(), StrategistConfig())

    research = next(task for task in tasks if isinstance(task, Research))
    assert research.research_key == "unlock_targeting_computer"


async def test_all_in_drops_mining_and_strikes_enemies() -> None:
    ctx = await _base_context()
    await ctx.world.observe_tile(
        TileObservation(q=5, r=5, has_resource=True, resource_type="titanium_ore"), cycle=1
    )
    await ctx.tracks.observe(Sighting(SightingSource.SCAN, 9, 9, 5))
    tasks = generate_tasks(Doctrine.ALL_IN, ctx, _assessment(), StrategistConfig())
    assert not any(isinstance(t, MineLoop) for t in tasks)
    assert any(isinstance(t, Strike) for t in tasks)


async def test_defend_keeps_mining_and_strikes() -> None:
    ctx = await _base_context()
    await ctx.world.observe_tile(
        TileObservation(q=5, r=5, has_resource=True, resource_type="titanium_ore"), cycle=1
    )
    await ctx.tracks.observe(Sighting(SightingSource.SCAN, 3, 0, 5))
    tasks = generate_tasks(Doctrine.DEFEND, ctx, _assessment(), StrategistConfig())
    assert any(isinstance(t, MineLoop) for t in tasks)
    assert any(isinstance(t, Strike) for t in tasks)


async def test_no_known_ore_generates_bootstrap_scouting() -> None:
    ctx = await _base_context()  # no resource nodes observed
    tasks = generate_tasks(Doctrine.EXPAND, ctx, _assessment(resource_nodes=0), StrategistConfig())
    scout = [t for t in tasks if t.kind == "scout_sector"]
    assert len(scout) == 1
    assert scout[0].max_assignees == 2
    # And it disappears once a node is known.
    await ctx.world.observe_tile(
        TileObservation(q=5, r=5, has_resource=True, resource_type="titanium_ore"), cycle=1
    )
    tasks = generate_tasks(Doctrine.EXPAND, ctx, _assessment(resource_nodes=1), StrategistConfig())
    assert not any(t.kind == "scout_sector" for t in tasks)
    assert any(isinstance(t, MineLoop) for t in tasks)


# --- fitness ---

def test_default_fitness_prefers_closer_drone() -> None:
    task = MineLoop(task_id="m", resource_node=(10, 0), dropoff_id="cc")

    @dataclass
    class D:
        coord: tuple[int, int]

    near = default_fitness(D((9, 0)), task)
    far = default_fitness(D((0, 0)), task)
    assert near > far


# --- tick integration ---

async def test_tick_allocates_and_registers_controller_output() -> None:
    ctx = await _base_context()
    await ctx.world.observe_tile(
        TileObservation(q=3, r=0, has_resource=True, resource_type="titanium_ore"), cycle=1
    )
    await ctx.world.upsert_drone("d1", q=0, r=0, cycle=1)
    await ctx.world.upsert_drone("d2", q=1, r=0, cycle=1)

    built: list[str] = []

    def mine_factory(entity, task, context):
        built.append(task.task_id)
        return [{"action": "noop", "payload": {}}]

    strat = Strategist(
        score_source=lambda: ScoreState(own=0, best_enemy=0),
        controller_factories={"mine_loop": mine_factory},
        config=StrategistConfig(),
    )
    await strat.start(ctx)
    try:
        doctrine = await strat.tick()
    finally:
        await strat.stop()

    # Even score, ready economy, no enemies -> EXPAND; a miner is assigned and
    # its controller output registered with the pipeliner.
    assert doctrine is Doctrine.EXPAND
    assert ctx.pipeliner.registered  # at least one drone got a plan
    assert built  # the mine controller factory was invoked


async def test_tick_scoreboard_shifts_posture_to_all_in() -> None:
    ctx = await _base_context()
    await ctx.world.upsert_drone("d1", q=0, r=0, cycle=1)
    await ctx.world.upsert_drone("d2", q=1, r=0, cycle=1)
    await ctx.tracks.observe(Sighting(SightingSource.SCAN, 40, 40, 5))  # enemy far (not near base)

    strat = Strategist(
        score_source=lambda: ScoreState(own=0, best_enemy=200),  # far behind
        config=StrategistConfig(aggro_margin=50),
    )
    await strat.start(ctx)
    try:
        doctrine = await strat.tick()
    finally:
        await strat.stop()
    assert doctrine is Doctrine.ALL_IN
