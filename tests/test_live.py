"""Tests for the live strategist wiring (scoreboard source, miner adapter,
controller factories)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.planning.controllers.miner import MinerPhase, MinerPlan, RoutedAction
from agent.planning.live import (
    ScoreboardSource,
    _MinerOutput,
    build_controller_factories,
    build_strategist,
)
from agent.planning.pipeliner import EntityKind, PlannedAction
from agent.planning.strategist import ScoreState, Strategist
from agent.planning.tasks import ProduceDrone, Research


@dataclass
class FakeResult:
    data: dict[str, Any]
    ok: bool = True
    status: int = 200


class FakeRest:
    def __init__(self, results: list[FakeResult]) -> None:
        self.results = results
        self.calls = 0

    async def get(self, path: str, params: dict[str, Any] | None = None) -> FakeResult:
        assert path == "/auth/scoreboard"
        self.calls += 1
        return self.results[min(self.calls - 1, len(self.results) - 1)]


SCOREBOARD = {
    "scoreboard": [
        {"user_id": "me", "team": 1, "score": 40},
        {"user_id": "them", "team": 2, "score": 90},
    ],
    "team_scores": {"1": 40, "2": 90, "3": 10, "0": 999},
}


# --- ScoreboardSource ---

async def test_scoreboard_source_resolves_team_and_best_enemy() -> None:
    src = ScoreboardSource(FakeRest([FakeResult(SCOREBOARD)]), "me", min_interval_s=0.0)
    state = await src()
    assert state == ScoreState(own=40, best_enemy=90)  # admin team 0 excluded


async def test_scoreboard_source_caches_within_interval() -> None:
    rest = FakeRest([FakeResult(SCOREBOARD)])
    src = ScoreboardSource(rest, "me", min_interval_s=60.0)
    await src()
    await src()
    await src()
    assert rest.calls == 1


async def test_scoreboard_source_serves_cache_on_failure() -> None:
    rest = FakeRest([FakeResult(SCOREBOARD), FakeResult({}, ok=False, status=503)])
    src = ScoreboardSource(rest, "me", min_interval_s=0.0)
    first = await src()
    second = await src()  # failure -> cached value
    assert second == first


async def test_scoreboard_source_feeds_strategist_read_score() -> None:
    src = ScoreboardSource(FakeRest([FakeResult(SCOREBOARD)]), "me", min_interval_s=0.0)
    strat = Strategist(score_source=src)
    state = await strat._read_score()
    assert state.lead == -50


# --- _MinerOutput ---

def _drone_leg(drone_id: str, n: int) -> MinerPlan:
    actions = tuple(
        RoutedAction(EntityKind.DRONE, drone_id, PlannedAction(f"propulsion/drive", {"i": i}))
        for i in range(n)
    )
    return MinerPlan(drone_id, MinerPhase.SEEK_RESOURCE, actions)


def _building_leg(drone_id: str, building_id: str) -> MinerPlan:
    actions = (
        RoutedAction(
            EntityKind.BUILDING, building_id, PlannedAction("charging_station/charge", {})
        ),
    )
    return MinerPlan(drone_id, MinerPhase.SEEK_CHARGER, actions)


class FakeMiner:
    def __init__(self, plans: list[MinerPlan]) -> None:
        self.drone_id = plans[0].drone_id if plans else "d1"
        self._plans = plans
        self.calls = 0

    def plan(self) -> MinerPlan:
        self.calls += 1
        return self._plans[min(self.calls - 1, len(self._plans) - 1)]


class FakePipeliner:
    def __init__(self) -> None:
        self.replaced: list[tuple[Any, str, list[Any]]] = []

    async def replace_plan(self, kind: Any, entity_id: str, output: Any) -> None:
        self.replaced.append((kind, entity_id, list(output)))


async def test_miner_output_buffers_drone_leg() -> None:
    out = _MinerOutput(FakeMiner([_drone_leg("d1", 3)]), FakePipeliner())
    a1, a2, a3 = await out(), await out(), await out()
    assert [a.payload["i"] for a in (a1, a2, a3)] == [0, 1, 2]


async def test_miner_output_replans_when_buffer_empty() -> None:
    miner = FakeMiner([_drone_leg("d1", 1), _drone_leg("d1", 1)])
    out = _MinerOutput(miner, FakePipeliner())
    await out()
    await out()
    assert miner.calls == 2


async def test_miner_output_waiting_returns_none() -> None:
    plan = MinerPlan("d1", MinerPhase.WAITING, (), reason="no state yet")
    out = _MinerOutput(FakeMiner([plan]), FakePipeliner())
    assert await out() is None


async def test_miner_output_routes_building_leg_once() -> None:
    pipeliner = FakePipeliner()
    miner = FakeMiner([_building_leg("d1", "charger")])
    out = _MinerOutput(miner, pipeliner)
    assert await out() is None
    assert await out() is None  # identical leg not reinstalled
    assert len(pipeliner.replaced) == 1
    kind, entity_id, actions = pipeliner.replaced[0]
    assert kind is EntityKind.BUILDING and entity_id == "charger"
    assert actions[0].action == "charging_station/charge"


# --- factories ---

@dataclass
class _BuildingState:
    building_id: str
    stored_resources: dict[str, int] = field(default_factory=dict)


class _Sim:
    """Mirrors the EntitySim surface live.py touches: research_tiers is a
    *property* (the live-run bug a method-shaped double previously masked) and
    get_building returns checkpointed stored resources."""

    def __init__(self, buildings: dict[str, dict[str, int]] | None = None) -> None:
        self._buildings = {
            bid: _BuildingState(bid, dict(stored)) for bid, stored in (buildings or {}).items()
        }

    @property
    def research_tiers(self) -> dict[str, int]:
        return {}

    def get_building(self, building_id: str) -> _BuildingState | None:
        return self._buildings.get(building_id)


@dataclass
class _Ctx:
    entity_sim: Any = field(default_factory=_Sim)
    pipeliner: Any = field(default_factory=FakePipeliner)
    world: Any = None
    threat_map: Any = None


@dataclass
class _Building:
    building_id: str


def test_factories_cover_strategist_task_kinds() -> None:
    factories = build_controller_factories()
    assert set(factories) == {"mine_loop", "research", "produce_drone", "scout_sector"}


def test_bootstrap_scout_alternates_scan_and_turn_forever() -> None:
    from agent.planning.tasks import ScoutSector

    factories = build_controller_factories()
    output = factories["scout_sector"](
        None, ScoutSector(task_id="scout:bootstrap", center=(0, 0), radius=8), _Ctx()
    )
    actions = [output() for _ in range(5)]
    assert [a.action for a in actions] == [
        "sensors/scan", "propulsion/turn", "sensors/scan", "propulsion/turn", "sensors/scan",
    ]
    assert actions[1].payload == {"direction": 1, "level": 1}


def test_research_factory_emits_when_affordable() -> None:
    factories = build_controller_factories()
    ctx = _Ctx(entity_sim=_Sim({"cc": {"enriched_unobtanium": 4}}))
    output = factories["research"](
        _Building("cc"), Research(task_id="research:unlock_auto_cannon",
                                  research_key="unlock_auto_cannon"), ctx
    )
    action = output()
    assert action.action == "command_center/research"
    assert action.payload["equipment_type"] == "unlock_auto_cannon"


def test_research_factory_idles_when_unaffordable() -> None:
    # 0 EU stored (and unknown stock): no action -> no live 'Insufficient
    # enriched_unobtanium' error spam (observed before this gate existed).
    factories = build_controller_factories()
    task = Research(task_id="research:unlock_auto_cannon", research_key="unlock_auto_cannon")
    broke = factories["research"](
        _Building("cc"), task, _Ctx(entity_sim=_Sim({"cc": {"enriched_unobtanium": 0}}))
    )
    unknown = factories["research"](_Building("cc"), task, _Ctx(entity_sim=_Sim()))
    assert broke() is None
    assert unknown() is None


def test_produce_factory_gates_then_exhausts() -> None:
    factories = build_controller_factories()
    ctx = _Ctx(
        entity_sim=_Sim({"plant": {"titanium_parts": 30, "battery_materials": 15}})
    )
    output = factories["produce_drone"](
        _Building("plant"), ProduceDrone(task_id="produce:miner:1"), ctx
    )
    first = output()
    assert first.action == "production/manufacture"
    assert output() is None  # one-shot plan exhausted


def test_produce_factory_idles_without_materials() -> None:
    factories = build_controller_factories()
    output = factories["produce_drone"](
        _Building("plant"), ProduceDrone(task_id="produce:miner:1"),
        _Ctx(entity_sim=_Sim({"plant": {"titanium_parts": 5}})),
    )
    assert output() is None


def test_build_strategist_wires_everything() -> None:
    strat = build_strategist(FakeRest([FakeResult(SCOREBOARD)]), "me")
    assert isinstance(strat, Strategist)
    assert set(strat.controller_factories) == {
        "mine_loop", "research", "produce_drone", "scout_sector",
    }
    assert isinstance(strat.score_source, ScoreboardSource)
