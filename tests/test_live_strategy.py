from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.bus import EventBus
from agent.config import Config
from agent.planning.allocator import TaskAllocator
from agent.planning.match_loop import LiveMatchStrategy
from agent.planning.pipeliner import PipelineStatus
from agent.sim.entity_sim import EntitySim
from agent.transport.action_tracker import EntityKind
from agent.world import ThreatMap, TileObservation, TrackStore, WorldModel
from agent.world.tiles import Terrain


@dataclass
class Result:
    data: dict[str, Any]
    ok: bool = True
    status: int = 200
    error_message: str = ""


class Rest:
    def __init__(self) -> None:
        self.queues = {
            "/queue/drones": [{"action_id": "stale-drone"}],
            "/queue/buildings": [{"action_id": "stale-building"}],
        }
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.deletes: list[str] = []

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Result:
        del params
        if path in self.queues:
            return Result({"actions": list(self.queues[path])})
        if path == "/drones":
            return Result({"drone_ids": ["d1", "d2"]})
        if path == "/buildings":
            return Result({"building_ids": ["cc", "refiner", "charger", "plant"]})
        if path == "/auth/me":
            return Result({"user_id": "us", "team": 1, "score": 10})
        if path == "/auth/scoreboard":
            return Result(
                {
                    "scoreboard": [
                        {"user_id": "us", "team": 1, "score": 10},
                        {"user_id": "them", "team": 2, "score": 20},
                    ],
                    "team_scores": {"1": 10, "2": 20},
                }
            )
        if path == "/game/status":
            return Result({"match_state": "active"})
        raise AssertionError(f"unexpected GET {path}")

    async def post(self, path: str, json: dict[str, Any] | None = None) -> Result:
        self.posts.append((path, dict(json or {})))
        return Result({"action_id": f"bootstrap-{len(self.posts)}"}, status=201)

    async def delete(self, path: str, json: dict[str, Any] | None = None) -> Result:
        del json
        self.deletes.append(path)
        for queue in self.queues.values():
            queue[:] = [action for action in queue if action["action_id"] not in path]
        return Result({})


class Pipeliner:
    def __init__(self) -> None:
        self.outputs: dict[tuple[EntityKind, str], Any] = {}
        self.statuses: dict[tuple[EntityKind, str], PipelineStatus] = {}
        self.replaced: list[tuple[EntityKind, str]] = []
        self.invalidated: list[tuple[EntityKind, str]] = []

    def register(self, kind: EntityKind | str, entity_id: str, output: Any) -> None:
        key = (EntityKind(kind), entity_id)
        assert key not in self.outputs
        self.outputs[key] = output
        self.statuses[key] = PipelineStatus(key[0], entity_id, 1, 1, 0, False, False)

    def status(self, kind: EntityKind | str, entity_id: str) -> PipelineStatus | None:
        return self.statuses.get((EntityKind(kind), entity_id))

    async def replace_plan(self, kind: EntityKind | str, entity_id: str, output: Any) -> None:
        key = (EntityKind(kind), entity_id)
        self.replaced.append(key)
        self.outputs[key] = output
        old = self.statuses[key]
        self.statuses[key] = PipelineStatus(
            key[0], entity_id, old.generation + 1, 1, 0, False, False
        )

    async def invalidate(
        self, kind: EntityKind | str, entity_id: str, replacement: Any = None
    ) -> None:
        del replacement
        key = (EntityKind(kind), entity_id)
        self.invalidated.append(key)
        old = self.statuses[key]
        self.statuses[key] = PipelineStatus(
            key[0], entity_id, old.generation + 1, 0, 0, True, True
        )

    def finish(self, kind: EntityKind, entity_id: str) -> None:
        old = self.statuses[(kind, entity_id)]
        self.statuses[(kind, entity_id)] = PipelineStatus(
            kind, entity_id, old.generation, 0, 0, True, False
        )


async def context() -> tuple[SimpleNamespace, Pipeliner]:
    bus = EventBus()
    world = WorldModel(bus)
    tracks = TrackStore(bus)
    sim = EntitySim()
    pipeline = Pipeliner()
    buildings = (
        ("cc", "command_center", (-5, 0)),
        ("refiner", "refiner", (0, 0)),
        ("charger", "drone_charging_station", (-2, 0)),
        ("plant", "drone_production_plant", (-4, 0)),
    )
    for building_id, building_type, origin in buildings:
        await world.upsert_building(
            building_id,
            building_type=building_type,
            origin=origin,
            tiles=(origin,),
            cycle=1,
        )
        sim.seed_building(building_id, building_type=building_type)
    await world.observe_tiles(
        [
            TileObservation(q=1, r=0, terrain_type=Terrain.NORMAL),
            TileObservation(
                q=2,
                r=0,
                terrain_type=Terrain.NORMAL,
                has_resource=True,
                resource_type="titanium_ore",
                resource_amount=100,
            ),
        ],
        cycle=1,
    )
    for drone_id, coord in (("d1", (1, 0)), ("d2", (1, 1))):
        await world.upsert_drone(
            drone_id, q=coord[0], r=coord[1], direction=0, cycle=1
        )
        sim.seed_drone(
            drone_id,
            current_battery=1000,
            max_battery=1000,
            hopper_capacity=20,
            equipment=("propulsion", "sensors", "drill", "hopper"),
        )
    return (
        SimpleNamespace(
            bus=bus,
            world=world,
            tracks=tracks,
            threat_map=ThreatMap(world, tracks),
            entity_sim=sim,
            allocator=TaskAllocator(),
            pipeliner=pipeline,
        ),
        pipeline,
    )


async def test_bootstrap_allocation_queue_transition_and_shutdown(tmp_path: Path) -> None:
    ctx, pipeline = await context()
    rest = Rest()
    cfg = Config(
        "https://game.test",
        "pilot",
        "secret",
        strategist_tick_s=60,
        scoreboard_poll_s=60,
        telemetry_path=tmp_path / "live.jsonl",
        world_database=tmp_path / "world.sqlite",
    )
    strategy = LiveMatchStrategy(rest, cfg, bootstrap_timeout_s=0.1)

    await strategy.start(ctx)
    report = await strategy.wait_ready(timeout_s=1)
    await strategy.tick()

    assert report.complete
    assert report.stale_actions_cleared == 2
    assert rest.deletes == [
        "/queue/drones/stale-drone",
        "/queue/buildings/stale-building",
    ]
    assert "/buildings/command_centre/status_report" in {path for path, _ in rest.posts}
    assert (EntityKind.DRONE, "d1") in pipeline.outputs
    assert (EntityKind.DRONE, "d2") in pipeline.outputs
    assert strategy.last_inputs is not None
    assert strategy.last_inputs.score == strategy._score
    assert strategy.last_inputs.economy.drone_cargo == {"d1": {}, "d2": {}}

    # Finish d1's mining leg while it is beside the refiner and give it cargo.
    # The next planner round must release its drone queue before claiming the
    # building queue for the unload leg.
    miner_id = next(
        entity_id
        for entity_id, assignment in ctx.allocator.last_result.assignments.items()
        if assignment.task.kind == "mine_loop"
    )
    pipeline.finish(EntityKind.DRONE, miner_id)
    ctx.entity_sim.seed_drone(
        miner_id,
        current_battery=900,
        max_battery=1000,
        cargo={"titanium_ore": 20},
        hopper_capacity=20,
        equipment=("propulsion", "sensors", "drill", "hopper"),
    )
    await ctx.world.upsert_drone(miner_id, q=1, r=0, direction=0, cycle=2)
    await strategy.tick()

    assert (EntityKind.DRONE, miner_id) in pipeline.invalidated
    assert (EntityKind.BUILDING, "refiner") in pipeline.outputs

    installed_keys = set(pipeline.outputs)
    await strategy.stop()

    assert installed_keys <= set(pipeline.invalidated)
    assert strategy._task is None
