"""Live wiring: install the Strategist into the composed runtime (E5.9).

E5.7 delivered the Strategist with controllers injected as factories; this
module supplies the *real* factories and score source so ``compose_runtime``
can run a playable agent:

- :class:`ScoreboardSource` — cached async reader of the public
  ``GET /auth/scoreboard`` that maps team totals into a :class:`ScoreState`.
- ``mine_loop`` factory — wraps :class:`MinerController` (a replanning
  controller) as an inexhaustible pipeliner ``ActionFactory`` for the drone's
  queue. Legs routed to a *building* queue (e.g. a charging-station charge)
  are installed on that building's pipeline as a one-shot plan.
- ``research`` factory — :class:`ResearchController` fed by the entity sim's
  live research tiers; its ``output`` is already an inexhaustible factory.
- ``produce_drone`` factory — :class:`ProductionController` expanded into a
  finite one-shot plan (the strategist re-issues it when the fleet changes).
- ``scout_sector`` factory — :class:`ScoutController` adapted into a buffered
  frontier leg that waits for each scan observation before replanning.
- :func:`build_strategist` — the one call ``__main__`` makes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agent.planning.controllers.base import BuildOrder, ProductionController, ResearchController
from agent.planning.controllers.charger import ChargerController
from agent.planning.controllers.miner import MinerController, MinerPhase
from agent.planning.controllers.scout import ScoutController, ScoutPhase
from agent.planning.pipeliner import EntityKind, PlannedAction
from agent.planning.strategist import ScoreState, Strategist, StrategistConfig
from agent.planning.tasks import MineLoop, ProduceDrone, Recharge, Research, ScoutSector, Task
from agent.rules.costs import DRONE_COST
from agent.rules.economy import UNLOCK_RULES

log = logging.getLogger("agent.planning.live")

# EU cost per research tier for non-unlock upgrades / baseline_drone.
_UPGRADE_TIER_EU = {1: 1, 2: 3, 3: 5}
_BASELINE_TIER_EU = {1: 3, 2: 4, 3: 6}


def _install_routed_building_plan(
    pipeliner: Any,
    kind: EntityKind,
    entity_id: str,
    actions: list[PlannedAction],
) -> asyncio.Task[Any] | None:
    """Register now or defer replacement until the current fill releases its lock.

    Controller factories are invoked from inside ``Pipeliner._fill``. Awaiting
    ``replace_plan`` there tries to reacquire the same lock and deadlocks the
    entire queue maintainer, so an existing building plan is switched in a
    separate event-loop task.
    """
    status_fn = getattr(pipeliner, "status", None)
    if callable(status_fn) and status_fn(kind, entity_id) is None:
        pipeliner.register(kind, entity_id, actions)
        return None

    task = asyncio.create_task(
        pipeliner.replace_plan(kind, entity_id, actions),
        name=f"route-{kind.value}-{entity_id}",
    )

    def report_failure(done: asyncio.Task[Any]) -> None:
        if done.cancelled():
            return
        error = done.exception()
        if error is not None:
            log.error(
                "failed installing routed building plan on %s %s",
                kind.value,
                entity_id,
                exc_info=(type(error), error, error.__traceback__),
            )

    task.add_done_callback(report_failure)
    return task


def _research_cost(research_key: str, target_level: int) -> dict[str, int]:
    rule = UNLOCK_RULES.get(research_key)
    if rule is not None:
        return dict(rule.cost)
    tiers = _BASELINE_TIER_EU if research_key == "baseline_drone" else _UPGRADE_TIER_EU
    return {"enriched_unobtanium": tiers.get(target_level, tiers[1])}


def _affordable(stored: dict[str, int] | None, cost: dict[str, int]) -> bool:
    """Whether a building's *known* stock covers ``cost``. Unknown stock counts
    as unaffordable — submitting blind is what floods the server with
    'Insufficient …' errors (observed live) and drowns our 100-cap inbox."""
    if stored is None:
        return False
    return all(stored.get(resource, 0) >= amount for resource, amount in cost.items())


class _AffordabilityGate:
    """Wrap a pipeliner factory so it emits only while the building can pay."""

    def __init__(self, inner: Any, sim: Any, building_id: str, cost: dict[str, int]) -> None:
        self._inner = inner
        self._sim = sim
        self._building_id = building_id
        self._cost = cost

    def __call__(self) -> PlannedAction | None:
        state = self._sim.get_building(self._building_id)
        stored = dict(state.stored_resources) if state is not None else None
        if not _affordable(stored, self._cost):
            return None
        return self._inner()


class ScoreboardSource:
    """Async callable returning the latest :class:`ScoreState`.

    Polls the public scoreboard at most once per ``min_interval_s`` (strategist
    ticks faster than the scoreboard meaningfully changes); serves the cached
    value otherwise, and keeps serving the last good value through fetch
    failures. Team identity is resolved from ``own_user_id`` on first sight.
    """

    def __init__(self, rest: Any, own_user_id: str, *, min_interval_s: float = 5.0) -> None:
        self._rest = rest
        self._own_user_id = own_user_id
        self._min_interval = min_interval_s
        self._own_team: int | None = None
        self._cached = ScoreState()
        self._fetched_at = 0.0

    async def __call__(self) -> ScoreState:
        now = time.monotonic()
        if now - self._fetched_at < self._min_interval:
            return self._cached
        self._fetched_at = now  # even on failure: don't hammer a broken endpoint
        result = await self._rest.get("/auth/scoreboard")
        if not result.ok:
            log.debug("scoreboard fetch failed (%s); serving cached score", result.status)
            return self._cached
        self._cached = self._parse(result.data)
        return self._cached

    def _parse(self, data: dict[str, Any]) -> ScoreState:
        if self._own_team is None:
            for entry in data.get("scoreboard", []) or []:
                if entry.get("user_id") == self._own_user_id:
                    self._own_team = int(entry.get("team", 0))
                    break
        team_scores = data.get("team_scores", {}) or {}
        own = 0
        best_enemy = 0
        for team, score in team_scores.items():
            try:
                team_number = int(team)
                value = int(score)
            except (TypeError, ValueError):
                continue
            if self._own_team is not None and team_number == self._own_team:
                own = value
            elif team_number != 0:  # team 0 = admins
                best_enemy = max(best_enemy, value)
        return ScoreState(own=own, best_enemy=best_enemy)


class _MinerOutput:
    """Adapt the replanning :class:`MinerController` to a pipeliner factory.

    Each call returns the next action of the current leg for the *drone's*
    queue, replanning from live world/sim state when the buffer runs out. A leg
    that targets a building queue (charging-station charge, dropoff-side work)
    is installed on that building's pipeline instead — deduplicated by leg
    signature so an unchanged building leg isn't reinstalled every call.
    """

    def __init__(self, controller: MinerController, pipeliner: Any) -> None:
        self._controller = controller
        self._pipeliner = pipeliner
        self._buffer: list[PlannedAction] = []
        self._last_building_leg: tuple[Any, ...] | None = None
        self._building_install: asyncio.Task[Any] | None = None

    async def __call__(self) -> PlannedAction | None:
        if self._buffer:
            return self._buffer.pop(0)

        plan = self._controller.plan()
        if plan.phase in (MinerPhase.WAITING, MinerPhase.BLOCKED):
            if plan.reason:
                log.debug("miner %s idle (%s): %s", self._controller.drone_id, plan.phase.value, plan.reason)
            return None
        key = plan.entity_key
        if key is None:
            return None
        kind, entity_id = key

        if kind is EntityKind.DRONE:
            self._buffer = list(plan.actions_for(EntityKind.DRONE, entity_id))
            self._last_building_leg = None
            return self._buffer.pop(0) if self._buffer else None

        # Building-owned leg: hand it to that building's queue as a one-shot
        # plan and report "nothing for the drone right now".
        actions = plan.actions_for(kind, entity_id)
        signature = (kind, entity_id, tuple((a.action, tuple(sorted(a.payload.items()))) for a in actions))
        if signature != self._last_building_leg:
            self._last_building_leg = signature
            try:
                self._building_install = _install_routed_building_plan(
                    self._pipeliner,
                    kind,
                    entity_id,
                    list(actions),
                )
            except Exception:
                log.exception("failed installing miner building leg on %s %s", kind.value, entity_id)
                self._last_building_leg = None
        return None


class _ScoutOutput:
    """Buffer one frontier leg and wait for its scan to update the world."""

    def __init__(self, controller: ScoutController, world: Any) -> None:
        self._controller = controller
        self._world = world
        self._buffer: list[PlannedAction] = []
        self._destination: tuple[int, int] | None = None
        self._awaiting_seen: int | None = None
        self._awaiting_cycle = 0

    def __call__(self) -> PlannedAction | None:
        if self._awaiting_seen is not None:
            assert self._destination is not None
            tile = self._world.get_tile(self._destination)
            last_seen = tile.last_seen if tile is not None else -1
            if last_seen <= self._awaiting_seen:
                # A failed/lost scan must not idle forever, but leave ample
                # time for the queued action and websocket report first.
                if self._world.cycle - self._awaiting_cycle < 20:
                    return None
            self._awaiting_seen = None

        if not self._buffer:
            plan = self._controller.plan()
            if plan.phase in (ScoutPhase.WAITING, ScoutPhase.BLOCKED):
                if plan.reason:
                    log.debug(
                        "scout %s idle (%s): %s",
                        self._controller.drone_id,
                        plan.phase.value,
                        plan.reason,
                    )
                return None
            self._buffer = list(plan.actions)
            self._destination = plan.destination

        action = self._buffer.pop(0) if self._buffer else None
        if action is not None and action.action == "sensors/scan":
            assert self._destination is not None
            tile = self._world.get_tile(self._destination)
            self._awaiting_seen = tile.last_seen if tile is not None else -1
            self._awaiting_cycle = self._world.cycle
        return action


class _RechargeOutput:
    """Serialize a charger route, then install its building-owned charge leg."""

    def __init__(self, controller: ChargerController, pipeliner: Any, world: Any) -> None:
        self._controller = controller
        self._pipeliner = pipeliner
        self._world = world
        self._buffer: list[PlannedAction] = []
        self._destination: tuple[int, int] | None = None
        self._awaiting_destination: tuple[int, int] | None = None
        self._charge_installed = False

    async def __call__(self) -> PlannedAction | None:
        if self._awaiting_destination is not None:
            drone = self._world.get_drone(self._controller.drone_id)
            if drone is None or drone.coord != self._awaiting_destination:
                return None
            self._awaiting_destination = None

        if self._buffer:
            action = self._buffer.pop(0)
            if not self._buffer:
                self._awaiting_destination = self._destination
            return action

        plan = self._controller.plan()
        if plan.phase in (MinerPhase.WAITING, MinerPhase.BLOCKED):
            return None
        key = plan.entity_key
        if key is None:
            return None
        kind, entity_id = key
        if kind is EntityKind.DRONE:
            self._buffer = list(plan.actions_for(kind, entity_id))
            self._destination = plan.destination
            if not self._buffer:
                return None
            action = self._buffer.pop(0)
            if not self._buffer:
                self._awaiting_destination = self._destination
            return action

        if not self._charge_installed:
            actions = list(plan.actions_for(kind, entity_id))
            _install_routed_building_plan(self._pipeliner, kind, entity_id, actions)
            self._charge_installed = True
        return None


def build_controller_factories() -> dict[str, Any]:
    """Task-kind -> factory mapping for :class:`Strategist`. Factories receive
    ``(entity, task, context)`` at call time, so they need no captured state."""

    def scout_sector(entity: Any, task: Task, context: Any) -> Any:
        assert isinstance(task, ScoutSector)
        controller = ScoutController(
            context.world,
            context.entity_sim,
            entity.drone_id,
            task,
            threat_cost=lambda coord: context.threat_map.cost_at(coord),
        )
        return _ScoutOutput(controller, context.world)

    def recharge(entity: Any, task: Task, context: Any) -> Any:
        assert isinstance(task, Recharge)
        controller = ChargerController(
            context.world,
            context.entity_sim,
            entity.drone_id,
            task,
            threat_cost=lambda coord: context.threat_map.cost_at(coord),
        )
        return _RechargeOutput(controller, context.pipeliner, context.world)

    def mine_loop(entity: Any, task: Task, context: Any) -> Any:
        assert isinstance(task, MineLoop)
        controller = MinerController(
            context.world,
            context.entity_sim,
            entity.drone_id,
            task,
            threat_cost=lambda coord: context.threat_map.cost_at(coord),
        )
        return _MinerOutput(controller, context.pipeliner)

    def research(entity: Any, task: Task, context: Any) -> Any:
        assert isinstance(task, Research)
        controller = ResearchController(
            entity.building_id,
            [task.research_key],
            lambda: context.entity_sim.research_tiers,  # property, not a method
        )
        cost = _research_cost(task.research_key, task.target_level or 1)
        return _AffordabilityGate(controller.output, context.entity_sim, entity.building_id, cost)

    def produce_drone(entity: Any, task: Task, context: Any) -> Any:
        assert isinstance(task, ProduceDrone)
        controller = ProductionController(entity.building_id, None)
        pending = list(controller.plan(BuildOrder(drones=1)).drone_actions)

        def next_action() -> PlannedAction | None:
            return pending.pop(0) if pending else None

        return _AffordabilityGate(
            next_action, context.entity_sim, entity.building_id, dict(DRONE_COST)
        )

    return {
        "mine_loop": mine_loop,
        "research": research,
        "produce_drone": produce_drone,
        "scout_sector": scout_sector,
        "recharge": recharge,
    }


def build_strategist(
    rest: Any,
    own_user_id: str,
    *,
    config: StrategistConfig | None = None,
    scoreboard_interval_s: float = 5.0,
) -> Strategist:
    """The playable-agent strategist ``__main__`` installs into the runtime."""
    return Strategist(
        score_source=ScoreboardSource(rest, own_user_id, min_interval_s=scoreboard_interval_s),
        controller_factories=build_controller_factories(),
        config=config or StrategistConfig(),
    )
