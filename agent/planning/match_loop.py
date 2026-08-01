"""Production bootstrap and autonomous match coordination."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from agent.commander.contract import SquadOrderDirective
from agent.commander.directives import (
    DirectiveState,
    DirectiveStore,
    TOPIC_DIRECTIVES_CHANGED,
)
from agent.config import Config
from agent.planning.controllers import (
    BuildOrder,
    ChargerController,
    FighterController,
    MinerController,
    ProductionController,
    RefinerController,
    ResearchController,
    ScoutController,
)
from agent.planning.live import _AffordabilityGate, _affordable, _research_cost
from agent.planning.pipeliner import ControllerOutput, PipelineError
from agent.planning.strategist import (
    Doctrine,
    ScoreState,
    StrategicAssessment,
    StrategistConfig,
    apply_commander_stance,
    assess,
    generate_tasks,
    next_doctrine,
)
from agent.planning.tasks import (
    MineLoop,
    ProduceDrone,
    Recharge,
    Research,
    ScoutSector,
    Strike,
    Task,
)
from agent.rules.costs import DRONE_COST
from agent.rules.hexmath import get_neighbor, hex_distance_cube
from agent.transport.action_tracker import (
    TOPIC_ACTION_LIFECYCLE,
    ActionLifecycleEvent,
    EntityKind,
    IntentState,
)
from agent.world.model import TOPIC_WORLD_CHANGED, BuildingRecord, DroneRecord, WorldChange
from agent.world.tracks import TOPIC_TRACK_CHANGED

log = logging.getLogger("agent.planning.live")

EntityKey = tuple[EntityKind, str]


class RestResult(Protocol):
    data: dict[str, Any]
    status: int
    error_message: str | None

    @property
    def ok(self) -> bool: ...


class RestClient(Protocol):
    async def get(self, path: str, params: dict[str, Any] | None = None) -> RestResult: ...

    async def post(self, path: str, json: dict[str, Any] | None = None) -> RestResult: ...

    async def delete(self, path: str, json: dict[str, Any] | None = None) -> RestResult: ...


@dataclass(frozen=True, slots=True)
class BootstrapReport:
    drone_ids: tuple[str, ...]
    building_ids: tuple[str, ...]
    stale_actions_cleared: int
    complete: bool


@dataclass(frozen=True, slots=True)
class EconomyState:
    """Player-visible inventories passed alongside strategist assessment."""

    building_inventories: Mapping[str, Mapping[str, int]]
    drone_cargo: Mapping[str, Mapping[str, int]]


@dataclass(frozen=True, slots=True)
class LiveStrategicInputs:
    score: ScoreState
    economy: EconomyState
    assessment: StrategicAssessment


@dataclass(slots=True)
class _InstalledPlan:
    owner: str
    task_id: str
    key: EntityKey


@dataclass(slots=True)
class LiveMatchStrategy:
    """Concrete planning strategy used by the production CLI.

    It deliberately owns queue keys, not just drones.  Miner legs may move from
    a drone queue to a refiner/charger queue; the ownership table serializes
    those transitions and every replacement increments the pipeliner generation.
    """

    rest: RestClient
    runtime_config: Config
    strategist_config: StrategistConfig | None = None
    bootstrap_timeout_s: float = 30.0
    doctrine: Doctrine = Doctrine.BUILD_UP
    commander_stance: str | None = field(default=None, init=False)
    squad_orders: Mapping[str, SquadOrderDirective] = field(
        default_factory=dict, init=False
    )
    bootstrap_report: BootstrapReport | None = field(default=None, init=False)
    last_inputs: LiveStrategicInputs | None = field(default=None, init=False)
    last_tasks: tuple[Task, ...] = field(default=(), init=False)
    _context: Any = field(default=None, init=False, repr=False)
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _stop: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _wake: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _ready: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _unsubscribers: list[Any] = field(default_factory=list, init=False, repr=False)
    _installed: dict[str, _InstalledPlan] = field(default_factory=dict, init=False, repr=False)
    _queue_owners: dict[EntityKey, str] = field(default_factory=dict, init=False, repr=False)
    _forced: set[str] = field(default_factory=set, init=False, repr=False)
    _directive_overrides: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _score: ScoreState = field(default_factory=ScoreState, init=False, repr=False)
    _user_id: str | None = field(default=None, init=False, repr=False)
    _team: int | None = field(default=None, init=False, repr=False)
    _next_score_poll: float = field(default=0.0, init=False, repr=False)
    _match_ended: bool = field(default=False, init=False, repr=False)
    _charging: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.bootstrap_timeout_s <= 0:
            raise ValueError("bootstrap_timeout_s must be positive")
        if self.strategist_config is None:
            self.strategist_config = StrategistConfig(
                tick_interval_s=self.runtime_config.strategist_tick_s
            )

    async def start(self, context: Any) -> None:
        if self._task is not None:
            return
        self._context = context
        self._stop.clear()
        self._wake.clear()
        self._ready.clear()
        directives = getattr(context, "directives", None)
        if isinstance(directives, DirectiveStore):
            self._apply_directive_state(directives.state)
        self._unsubscribers = [
            context.bus.subscribe(TOPIC_WORLD_CHANGED, self._on_world_change),
            context.bus.subscribe(TOPIC_TRACK_CHANGED, self._on_change),
            context.bus.subscribe(TOPIC_ACTION_LIFECYCLE, self._on_lifecycle),
            context.bus.subscribe(TOPIC_DIRECTIVES_CHANGED, self._on_directives_changed),
        ]
        self._task = asyncio.create_task(self._run(), name="live-match-strategy")
        await asyncio.sleep(0)

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        task = self._task
        if task is not None:
            try:
                await task
            finally:
                self._task = None
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()

    async def wait_ready(self, timeout_s: float | None = None) -> BootstrapReport:
        if timeout_s is None:
            await self._ready.wait()
        else:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout_s)
        if self.bootstrap_report is None:
            raise RuntimeError("live strategy stopped before bootstrap completed")
        return self.bootstrap_report

    async def wait(self, duration_s: float | None = None) -> None:
        task = self._task
        if task is None:
            raise RuntimeError("live strategy has not been started")
        if duration_s is None:
            await task
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=duration_s)
        except asyncio.TimeoutError:
            return

    async def tick(self) -> Doctrine:
        """Run one assessment, allocation, and controller-installation round."""

        context = self._require_context()
        await self._poll_public_state()
        score = self._score
        config = self.strategist_config
        assert config is not None
        assessment = assess(context, score, config)
        doctrine = apply_commander_stance(
            next_doctrine(self.doctrine, assessment), self.commander_stance
        )
        doctrine_changed = doctrine is not self.doctrine
        if doctrine_changed:
            log.info("doctrine %s -> %s (%s)", self.doctrine.value, doctrine.value, assessment)
            self._forced.update(self._installed)
        self.doctrine = doctrine
        self.last_inputs = LiveStrategicInputs(score, self._economy_state(), assessment)

        tasks = self._tasks(doctrine, assessment)
        drones = list(context.world.drones())
        recharge_tasks, safety_pins = self._recharge_work(drones)
        self.last_tasks = tuple(recharge_tasks + tasks)
        drone_tasks = recharge_tasks + [
            task
            for task in tasks
            if isinstance(task, (MineLoop, ScoutSector, Strike))
        ]
        # Commander directives apply to normal mission allocation, but battery
        # safety is non-negotiable and must pre-empt an override while a drone
        # is below its recovery threshold.
        allocation_pins = dict(self._directive_overrides)
        allocation_pins.update(safety_pins)
        result = context.allocator.allocate(
            drones,
            drone_tasks,
            self._fitness,
            pinned=allocation_pins,
        )
        assigned = set(result)

        for drone_id in list(self._installed):
            if drone_id.startswith("drone:") and drone_id[6:] not in assigned:
                await self._release(drone_id)

        for drone_id, task in result.items():
            await self._maintain_drone(drone_id, task)

        await self._maintain_base_tasks(tasks)
        await self._release_lost_entities()
        self._forced.clear()
        return doctrine

    async def _run(self) -> None:
        try:
            self.bootstrap_report = await self._bootstrap()
            self._ready.set()
            while not self._stop.is_set() and not self._match_ended:
                try:
                    await self.tick()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("live planning tick failed")
                self._wake.clear()
                interval = self.strategist_config.tick_interval_s  # type: ignore[union-attr]
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("live strategy failed during startup")
        finally:
            self._ready.set()
            await self._release_all()

    async def _bootstrap(self) -> BootstrapReport:
        context = self._require_context()
        stale = await self._clear_stale_queues()
        drone_ids = await self._owned_ids("/drones", "drone_ids", "drones", "drone_id")
        building_ids = await self._owned_ids(
            "/buildings", "building_ids", "buildings", "building_id"
        )

        await self._post_required(
            "/buildings/command_centre/status_report", {"efficiency": 1.0}
        )
        for drone_id in drone_ids:
            await self._post_required(f"/drones/{drone_id}/status", {})
            await self._post_required(f"/drones/{drone_id}/sensors/scan", {"level": 1})
        for building_id in building_ids:
            await self._post_required(
                f"/buildings/{building_id}/common/status", {"efficiency": 1.0}
            )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.bootstrap_timeout_s
        complete = False
        while not self._stop.is_set():
            world_drones = {
                drone.drone_id
                for drone in context.world.drones()
                if drone.coord is not None
            }
            sim_drones = set(context.entity_sim.drones)
            world_buildings = {building.building_id for building in context.world.buildings()}
            sim_buildings = set(context.entity_sim.buildings)
            complete = (
                set(drone_ids) <= world_drones
                and set(drone_ids) <= sim_drones
                and set(building_ids) <= world_buildings
                and set(building_ids) <= sim_buildings
                and context.world.tile_count > 0
            )
            if complete or loop.time() >= deadline:
                break
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=min(0.25, deadline - loop.time()))
            except asyncio.TimeoutError:
                pass
        if not complete:
            log.warning(
                "startup sweep timed out with %d/%d drones and %d/%d buildings checkpointed",
                len(set(drone_ids) & set(context.entity_sim.drones)),
                len(drone_ids),
                len(set(building_ids) & set(context.entity_sim.buildings)),
                len(building_ids),
            )
        else:
            log.info(
                "startup sweep completed for %d drones and %d buildings",
                len(drone_ids),
                len(building_ids),
            )
        return BootstrapReport(tuple(drone_ids), tuple(building_ids), stale, complete)

    async def _clear_stale_queues(self) -> int:
        cleared = 0
        for kind in EntityKind:
            result = await self.rest.get(kind.queue_path)
            if not result.ok:
                raise RuntimeError(f"could not resynchronise {kind.value} queue")
            actions = result.data.get("actions", [])
            if not isinstance(actions, list):
                raise RuntimeError(f"malformed {kind.value} queue response")
            for action in actions:
                if not isinstance(action, Mapping):
                    continue
                action_id = action.get("action_id") or action.get("id")
                if not isinstance(action_id, str) or not action_id:
                    continue
                deleted = await self.rest.delete(f"{kind.queue_path}/{action_id}")
                if not deleted.ok and getattr(deleted, "status", None) != 404:
                    raise RuntimeError(f"could not clear stale action {action_id}")
                cleared += 1
        return cleared

    async def _owned_ids(
        self,
        path: str,
        documented_key: str,
        legacy_key: str,
        item_key: str,
    ) -> list[str]:
        result = await self.rest.get(path)
        if not result.ok:
            raise RuntimeError(f"owned-asset discovery failed for {path}")
        raw = result.data.get(documented_key, result.data.get(legacy_key, []))
        if not isinstance(raw, list):
            raise RuntimeError(f"owned-asset discovery returned malformed {documented_key}")
        found: list[str] = []
        for item in raw:
            value = item.get(item_key, item.get("id")) if isinstance(item, Mapping) else item
            if isinstance(value, str) and value and value not in found:
                found.append(value)
        return found

    async def _post_required(self, path: str, payload: dict[str, Any]) -> None:
        result = await self.rest.post(path, json=payload)
        if not result.ok:
            detail = getattr(result, "error_message", None) or result.data
            raise RuntimeError(f"bootstrap POST {path} failed: {detail}")

    async def _poll_public_state(self) -> None:
        now = time.monotonic()
        if now < self._next_score_poll:
            return
        self._next_score_poll = now + self.runtime_config.scoreboard_poll_s

        me = await self.rest.get("/auth/me")
        if me.ok:
            self._user_id = _string(me.data.get("user_id")) or self._user_id
            self._team = _integer(me.data.get("team"), self._team)
        scoreboard = await self.rest.get("/auth/scoreboard")
        if scoreboard.ok:
            self._score = self._score_from(scoreboard.data, me.data if me.ok else {})
        game = await self.rest.get("/game/status")
        if game.ok:
            state = str(game.data.get("match_state", "active")).lower()
            self._match_ended = state == "ended"

    def _score_from(self, payload: Mapping[str, Any], me: Mapping[str, Any]) -> ScoreState:
        own = _integer(me.get("score"), 0) or 0
        best_enemy = 0
        rows = payload.get("scoreboard", [])
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                score = _integer(row.get("score"), 0) or 0
                user_id = _string(row.get("user_id"))
                team = _integer(row.get("team"), None)
                if self._user_id is not None and user_id == self._user_id:
                    own = score
                elif self._team is None or team != self._team:
                    best_enemy = max(best_enemy, score)
        team_scores = payload.get("team_scores")
        if isinstance(team_scores, Mapping) and self._team is not None:
            own = (
                _integer(team_scores.get(str(self._team), team_scores.get(self._team)), own)
                or own
            )
            enemy_scores = [
                _integer(value, 0) or 0
                for key, value in team_scores.items()
                if str(key) != str(self._team) and str(key) != "0"
            ]
            if enemy_scores:
                best_enemy = max(enemy_scores)
        return ScoreState(own=own, best_enemy=best_enemy)

    def _tasks(self, doctrine: Doctrine, assessment: StrategicAssessment) -> list[Task]:
        context = self._require_context()
        config = self.strategist_config
        assert config is not None
        generated = generate_tasks(doctrine, context, assessment, config)

        # Prefer a refiner as the miner drop-off so one completed mining cycle
        # naturally unlocks refining work.  Typed task replacement keeps all
        # other strategist policy intact.
        refiners = sorted(
            (
                building
                for building in context.world.buildings()
                if building.building_type == "refiner"
            ),
            key=lambda building: building.building_id,
        )
        if refiners:
            generated = [
                replace(task, dropoff_id=refiners[0].building_id)
                if isinstance(task, MineLoop)
                else task
                for task in generated
            ]

        drones = list(context.world.drones())
        base = self._base_origin()
        # Scouting remains present in every doctrine except a desperate all-in;
        # each sector has capacity one so both starter drones can receive work.
        if doctrine is not Doctrine.ALL_IN and drones:
            scout_count = max(1, len(drones) - sum(isinstance(t, MineLoop) for t in generated))
            for index in range(min(len(drones), scout_count)):
                center = base
                direction = index % 6
                for _ in range(8 + 4 * (index // 6)):
                    center = get_neighbor(*center, direction)
                generated.append(
                    ScoutSector(
                        task_id=f"scout:{direction}:{index // 6}",
                        center=center,
                        radius=4,
                        priority=1.2,
                    )
                )
        return generated

    def _recharge_work(
        self, drones: Sequence[DroneRecord]
    ) -> tuple[list[Recharge], dict[str, Recharge]]:
        context = self._require_context()
        config = self.strategist_config
        assert config is not None
        live_ids = {drone.drone_id for drone in drones}
        self._charging.intersection_update(live_ids)
        tasks: list[Recharge] = []
        pins: dict[str, Recharge] = {}
        for drone in drones:
            state = context.entity_sim.get_drone(drone.drone_id)
            if state is not None and state.max_battery > 0:
                fraction = state.current_battery / state.max_battery
                if fraction <= config.recharge_trigger_fraction:
                    self._charging.add(drone.drone_id)
                elif fraction >= config.recharge_recovery_fraction:
                    self._charging.discard(drone.drone_id)
            if drone.drone_id not in self._charging:
                continue
            task = Recharge(
                task_id=f"recharge:{drone.drone_id}",
                drone_id=drone.drone_id,
                target_fraction=config.recharge_recovery_fraction,
                priority=100.0,
            )
            tasks.append(task)
            pins[drone.drone_id] = task
        return tasks, pins

    def _fitness(self, entity: DroneRecord, task: Task) -> float | None:
        context = self._require_context()
        state = context.entity_sim.get_drone(entity.drone_id)
        if state is None:
            return None
        equipment = state.functional_equipment
        required: set[str]
        target: tuple[int, int] | None = None
        if isinstance(task, Recharge):
            return 1.0 if entity.drone_id == task.drone_id else None
        if isinstance(task, MineLoop):
            required = {"propulsion", "drill", "hopper"}
            target = task.resource_node
        elif isinstance(task, ScoutSector):
            required = {"propulsion", "sensors"}
            target = task.center
        elif isinstance(task, Strike):
            required = {"propulsion"}
            if not equipment.intersection(
                {"laser_cannon", "auto_cannon", "wire_guided_missile", "howitzer"}
            ):
                return None
            target = task.target if isinstance(task.target, tuple) else None
        else:
            return None
        if not required <= equipment:
            return None
        distance = (
            0
            if entity.coord is None or target is None
            else hex_distance_cube(*entity.coord, *target)
        )
        return float(task.priority) / (1.0 + distance)

    async def _maintain_drone(self, drone_id: str, task: Task) -> None:
        owner = f"drone:{drone_id}"
        current = self._installed.get(owner)
        forced = owner in self._forced or current is None or current.task_id != task.task_id
        if current is not None and not forced and not self._plan_finished(current):
            return
        try:
            key, output = self._controller_output(drone_id, task)
        except Exception:
            log.exception("controller failed to plan %s for %s", task.task_id, drone_id)
            return
        if key is None or not tuple(output):
            if current is not None and forced:
                await self._release(owner)
            return
        await self._install(owner, task.task_id, key, output, force=forced)

    def _controller_output(
        self, drone_id: str, task: Task
    ) -> tuple[EntityKey | None, tuple[Any, ...]]:
        context = self._require_context()
        if isinstance(task, MineLoop):
            plan = MinerController(
                context.world,
                context.entity_sim,
                drone_id,
                task,
                threat_cost=context.threat_map,
            ).plan()
            key = plan.entity_key
            return key, () if key is None else plan.actions_for(*key)
        if isinstance(task, Recharge):
            plan = ChargerController(
                context.world,
                context.entity_sim,
                drone_id,
                task,
                threat_cost=context.threat_map,
            ).plan()
            key = plan.entity_key
            return key, () if key is None else plan.actions_for(*key)
        if isinstance(task, ScoutSector):
            plan = ScoutController(
                context.world,
                context.entity_sim,
                drone_id,
                task,
                tracks=context.tracks,
                threat_cost=context.threat_map,
            ).plan()
            return (EntityKind.DRONE, drone_id), plan.actions
        if isinstance(task, Strike):
            plan = FighterController(
                context.world,
                context.tracks,
                context.entity_sim,
                drone_id,
                task,
                threat_cost=context.threat_map,
            ).plan()
            return (EntityKind.DRONE, drone_id), plan.actions
        return None, ()

    async def _maintain_base_tasks(self, tasks: Sequence[Task]) -> None:
        context = self._require_context()
        active: set[str] = set()
        for task in tasks:
            building = self._target_building(task)
            if building is None:
                continue
            owner = f"base:{building.building_id}"
            active.add(owner)
            current = self._installed.get(owner)
            forced = owner in self._forced or current is None or current.task_id != task.task_id
            if current is not None and not forced and not self._plan_finished(current):
                continue
            output = self._base_output(building, task)
            if output:
                await self._install(
                    owner,
                    task.task_id,
                    (EntityKind.BUILDING, building.building_id),
                    output,
                    force=forced,
                )

        # Refining is state-driven rather than a strategist task: run a bounded
        # batch whenever a refiner has ore and its queue is available.
        for building in context.world.buildings():
            if building.building_type != "refiner":
                continue
            owner = f"refiner:{building.building_id}"
            active.add(owner)
            current = self._installed.get(owner)
            if current is not None and not self._plan_finished(current):
                continue
            plan = RefinerController(building.building_id).plan(
                lambda building_id=building.building_id: self._building_inventory(building_id)
            )
            if not plan.empty:
                await self._install(
                    owner,
                    f"refine:{plan.ore_type}",
                    (EntityKind.BUILDING, building.building_id),
                    plan.actions,
                    force=False,
                )

        for owner in list(self._installed):
            if (owner.startswith("base:") or owner.startswith("refiner:")) and owner not in active:
                await self._release(owner)

    def _base_output(self, building: BuildingRecord, task: Task) -> ControllerOutput:
        context = self._require_context()
        if isinstance(task, Research):
            controller = ResearchController(
                building.building_id,
                (task,),
                lambda: context.entity_sim.research_tiers,
            )
            target = task.target_level or (1 if task.research_key.startswith("unlock_") else 3)

            def next_research_action() -> Any:
                current = context.entity_sim.research_tiers.get(task.research_key, 0)
                if current >= target:
                    return None
                state = context.entity_sim.get_building(building.building_id)
                stored = dict(state.stored_resources) if state is not None else None
                cost = _research_cost(task.research_key, min(target, current + 1))
                if not _affordable(stored, cost):
                    return None
                return controller.next_action()

            return next_research_action
        if isinstance(task, ProduceDrone):
            pending = list(
                ProductionController(building.building_id, None)
                .plan(BuildOrder(drones=1))
                .drone_actions
            )

            def next_production_action() -> Any:
                return pending.pop(0) if pending else None

            return _AffordabilityGate(
                next_production_action,
                context.entity_sim,
                building.building_id,
                dict(DRONE_COST),
            )
        return ()

    def _target_building(self, task: Task) -> BuildingRecord | None:
        context = self._require_context()
        buildings = list(context.world.buildings())
        if isinstance(task, ProduceDrone):
            return next(
                (b for b in buildings if b.building_type == "drone_production_plant"),
                None,
            )
        if isinstance(task, Research):
            return next((b for b in buildings if b.building_type == "command_center"), None)
        return None

    async def _install(
        self,
        owner: str,
        task_id: str,
        key: EntityKey,
        output: ControllerOutput,
        *,
        force: bool,
    ) -> bool:
        context = self._require_context()
        current = self._installed.get(owner)
        if current is not None and current.key != key:
            if not force and not self._plan_finished(current):
                return False
            # Release the owner's obsolete queue before checking whether a
            # shared destination queue is busy. A safety diversion must never
            # leave ordinary mission work running while the drone waits its
            # turn at a charging station.
            await self._release(owner)

        incumbent = self._queue_owners.get(key)
        if incumbent is not None and incumbent != owner:
            incumbent_plan = self._installed.get(incumbent)
            if incumbent_plan is not None and not self._plan_finished(incumbent_plan):
                return False
            await self._release(incumbent)

        status = context.pipeliner.status(*key)
        try:
            if status is None:
                context.pipeliner.register(*key, output)
            else:
                await context.pipeliner.replace_plan(*key, output)
        except PipelineError as exc:
            log.warning("could not install %s on %s %s: %s", task_id, key[0].value, key[1], exc)
            return False
        self._queue_owners[key] = owner
        self._installed[owner] = _InstalledPlan(owner, task_id, key)
        return True

    def _plan_finished(self, plan: _InstalledPlan) -> bool:
        status = self._require_context().pipeliner.status(*plan.key)
        return status is None or (status.exhausted and status.queue_depth == 0)

    async def _release(self, owner: str) -> None:
        plan = self._installed.pop(owner, None)
        if plan is None:
            return
        self._queue_owners.pop(plan.key, None)
        status = self._require_context().pipeliner.status(*plan.key)
        if status is None:
            return
        try:
            await self._require_context().pipeliner.invalidate(*plan.key, ())
        except PipelineError as exc:
            log.warning("could not release %s plan on %s: %s", owner, plan.key, exc)

    async def _release_all(self) -> None:
        for owner in list(self._installed):
            await self._release(owner)

    async def _release_lost_entities(self) -> None:
        context = self._require_context()
        drone_ids = {drone.drone_id for drone in context.world.drones()}
        building_ids = {building.building_id for building in context.world.buildings()}
        for owner, plan in list(self._installed.items()):
            kind, entity_id = plan.key
            if (kind is EntityKind.DRONE and entity_id not in drone_ids) or (
                kind is EntityKind.BUILDING and entity_id not in building_ids
            ):
                await self._release(owner)

    def _economy_state(self) -> EconomyState:
        context = self._require_context()
        buildings = {
            building_id: dict(state.stored_resources)
            for building_id, state in context.entity_sim.buildings.items()
        }
        drones = {
            drone_id: dict(state.cargo)
            for drone_id, state in context.entity_sim.drones.items()
        }
        return EconomyState(buildings, drones)

    def _building_inventory(self, building_id: str) -> Mapping[str, int]:
        state = self._require_context().entity_sim.get_building(building_id)
        return {} if state is None else state.stored_resources

    def _base_origin(self) -> tuple[int, int]:
        buildings = list(self._require_context().world.buildings())
        command = next((b for b in buildings if b.building_type == "command_center"), None)
        if command is not None and command.origin is not None:
            return command.origin
        for building in buildings:
            if building.origin is not None:
                return building.origin
        return (0, 0)

    async def _on_world_change(self, _topic: str, change: WorldChange) -> None:
        if change.kind == "removed":
            for key in change.keys:
                if isinstance(key, str):
                    self._forced.add(f"drone:{key}")
                    self._forced.add(f"base:{key}")
                    self._forced.add(f"refiner:{key}")
        self._wake.set()

    async def _on_change(self, _topic: str, _payload: Any) -> None:
        self._wake.set()

    async def _on_lifecycle(self, _topic: str, event: ActionLifecycleEvent) -> None:
        if event.state not in {IntentState.FAILED, IntentState.ABANDONED}:
            return
        owner = self._queue_owners.get((event.entity_kind, event.entity_id))
        if owner is not None:
            self._forced.add(owner)
        self._wake.set()

    async def _on_directives_changed(self, _topic: str, state: DirectiveState) -> None:
        previous_stance = self.commander_stance
        self._apply_directive_state(state)
        if self.commander_stance != previous_stance:
            self._forced.update(self._installed)
        self._wake.set()

    def _apply_directive_state(self, state: DirectiveState) -> None:
        self.commander_stance = None if state.stance is None else state.stance.stance
        self.squad_orders = state.squad_orders
        self._directive_overrides = dict(state.pinned_tasks)

    def _require_context(self) -> Any:
        if self._context is None:
            raise RuntimeError("live strategy has not been started")
        return self._context


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _integer(value: Any, fallback: int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


__all__ = [
    "BootstrapReport",
    "EconomyState",
    "LiveMatchStrategy",
    "LiveStrategicInputs",
]
