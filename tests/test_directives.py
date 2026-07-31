from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from agent.bus import EventBus
from agent.commander.api import CommanderAPI
from agent.commander.contract import (
    EntityOverrideDirective,
    SquadOrderDirective,
    StanceDirective,
)
from agent.commander.directives import DirectiveStore, TOPIC_DIRECTIVES_CHANGED
from agent.planning.allocator import TaskAllocator
from agent.planning.strategist import Doctrine, Strategist, apply_commander_stance
from agent.planning.tasks import Task
from agent.world.model import WorldModel
from agent.world.tracks import TrackStore


NOW = datetime(2026, 7, 31, 16, 0, tzinfo=timezone.utc)
NOW_JSON = "2026-07-31T16:00:00Z"


@dataclass
class Candidate:
    drone_id: str
    scores: dict[str, float]


def _score(candidate: Candidate, task: Task) -> float:
    return candidate.scores[task.task_id]


async def test_post_directives_validates_and_activates_all_contract_kinds() -> None:
    bus = EventBus()
    store = DirectiveStore(bus, clock=lambda: NOW)
    commander = CommanderAPI(WorldModel(bus), TrackStore(bus), bus, directives=store)
    transport = httpx.ASGITransport(app=commander.app)

    payloads = [
        {
            "kind": "stance",
            "directive_id": "stance-1",
            "issued_at": NOW_JSON,
            "ttl_seconds": 30,
            "stance": "defensive",
        },
        {
            "kind": "squad_order",
            "directive_id": "squad-1",
            "issued_at": NOW_JSON,
            "ttl_seconds": 30,
            "squad_id": "north",
            "order": "hold",
            "target": {"q": 4, "r": -2},
        },
        {
            "kind": "entity_override",
            "directive_id": "override-1",
            "issued_at": NOW_JSON,
            "ttl_seconds": 30,
            "entity_id": "drone-1",
            "task_id": "scout:north",
        },
    ]

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for payload in payloads:
            response = await client.post("/v1/directives", json=payload)
            assert response.status_code == 202
            assert response.json() == payload

        invalid = dict(payloads[0], ttl_seconds=0)
        assert (await client.post("/v1/directives", json=invalid)).status_code == 422

    assert store.state.stance == StanceDirective.model_validate(payloads[0])
    assert store.state.squad_orders == {
        "north": SquadOrderDirective.model_validate(payloads[1])
    }
    assert store.state.pinned_tasks == {"drone-1": "scout:north"}

    await commander.close()
    await store.close()


async def test_entity_override_survives_replanning_then_expires_to_autonomy() -> None:
    now = [NOW]
    bus = EventBus()
    store = DirectiveStore(bus, clock=lambda: now[0])
    states = []

    async def record(_topic: str, state) -> None:
        states.append(state)

    bus.subscribe(TOPIC_DIRECTIVES_CHANGED, record)
    override = EntityOverrideDirective(
        kind="entity_override",
        directive_id="override-1",
        issued_at=NOW,
        ttl_seconds=10,
        entity_id="drone-1",
        task_id="commander-task",
    )
    await store.apply(override)

    allocator = TaskAllocator(_score)
    drone = Candidate("drone-1", {"autonomous-task": 10.0, "commander-task": 0.0})
    autonomy = Task("autonomous-task")
    commander_task = Task("commander-task")

    first = allocator.allocate(
        [drone], [autonomy, commander_task], pinned=store.state.pinned_tasks
    )
    replan = allocator.allocate(
        [drone], [autonomy, commander_task], pinned=store.state.pinned_tasks
    )

    assert first["drone-1"] is commander_task
    assert replan["drone-1"] is commander_task
    assert replan.assignment_for("drone-1").pinned

    now[0] += timedelta(seconds=10)
    # A read at the deadline must neither expose the stale pin nor consume the
    # expiry change before it can be published to strategists.
    assert store.state.pinned_tasks == {}
    await store.refresh()
    autonomous = allocator.allocate(
        [drone], [autonomy, commander_task], pinned=store.state.pinned_tasks
    )

    assert store.state.pinned_tasks == {}
    assert autonomous["drone-1"] is autonomy
    assert not autonomous.assignment_for("drone-1").pinned
    assert len(states) == 2
    await store.close()


async def test_stance_change_propagates_to_strategist_and_maps_doctrine() -> None:
    bus = EventBus()
    store = DirectiveStore(bus, clock=lambda: NOW)
    strategist = Strategist()
    bus.subscribe(TOPIC_DIRECTIVES_CHANGED, strategist._on_directives_changed)

    await store.apply(
        StanceDirective(
            kind="stance",
            directive_id="stance-1",
            issued_at=NOW,
            ttl_seconds=30,
            stance="aggressive",
        )
    )

    assert strategist.commander_stance == "aggressive"
    assert apply_commander_stance(Doctrine.EXPAND, strategist.commander_stance) is Doctrine.ALL_IN
    assert apply_commander_stance(Doctrine.RAID, "balanced") is Doctrine.RAID
    assert apply_commander_stance(Doctrine.RAID, "defensive") is Doctrine.DEFEND
    assert apply_commander_stance(Doctrine.RAID, "retreat") is Doctrine.BUILD_UP
    await store.close()
