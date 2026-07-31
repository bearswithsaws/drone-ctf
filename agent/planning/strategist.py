"""Strategist: the doctrine FSM that turns the composed runtime into play.

Every planning primitive already exists — pathfinding, the pipeliner, the task
allocator, and the miner / base / fighter controllers. What was missing is the
layer that *decides*: which doctrine we are in, which tasks that implies, who
does them, and how the scoreboard should shift our risk posture. This module is
that layer, installed through the runtime's :class:`PlanningStrategy` boundary.

Design in three testable pieces:

1. **Assessment** (`assess`) — a pure read of the world, enemy tracks, threat
   map, and the latest score into a :class:`StrategicAssessment`.
2. **Doctrine FSM** (`next_doctrine`) — pure transition from the current
   doctrine + assessment to the next, with explicit triggers. Threat overrides
   everything (→ DEFEND); the score margin sets a posture (protect a lead vs
   force trades) that steers the rest.
3. **Task generation** (`generate_tasks`) — pure map from doctrine + assessment
   to the concrete :mod:`agent.planning.tasks` set for this planning round.

The :class:`Strategist` ties them together on a periodic tick: assess → pick
doctrine → generate tasks → allocate to drones → (re)register the matching
controller output with the pipeliner. Controllers are injected as factories so
the strategist stays unit-testable and the runtime supplies the real ones.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from agent.planning.pipeliner import ControllerOutput
from agent.planning.tasks import MineLoop, ProduceDrone, Research, ScoutSector, Strike, Task
from agent.rules import hexmath

log = logging.getLogger("agent.planning.strategist")

Coord = tuple[int, int]


class Doctrine(str, Enum):
    BUILD_UP = "build_up"   # establish economy + core research
    EXPAND = "expand"       # grow economy / map control from a safe base
    DEFEND = "defend"       # enemy near base — protect assets
    RAID = "raid"           # pressure enemy economy while even
    ALL_IN = "all_in"       # behind on score — force trades


class Posture(str, Enum):
    PROTECT = "protect"     # ahead — avoid risky trades
    NEUTRAL = "neutral"
    AGGRESSIVE = "aggressive"  # behind — seek trades


@dataclass(frozen=True, slots=True)
class ScoreState:
    """Latest scoreboard read (from public GET /auth/scoreboard polling)."""

    own: int = 0
    best_enemy: int = 0

    @property
    def lead(self) -> int:
        return self.own - self.best_enemy


@dataclass(frozen=True, slots=True)
class StrategistConfig:
    protect_margin: int = 50        # lead >= this -> PROTECT
    aggro_margin: int = 50          # deficit >= this -> AGGRESSIVE
    min_economy_drones: int = 2     # drones needed before we call economy "ready"
    drone_target: int = 4           # produce drones until we have this many
    base_defense_radius: int = 8    # tiles from any own building counts as "near base"
    tick_interval_s: float = 1.5
    # Ordered economy-first research plan the strategist walks through.
    research_order: tuple[str, ...] = (
        "unlock_auto_cannon",
        "baseline_drone",
        "unlock_targeting_computer",
        "unlock_shield_generator",
    )


@dataclass(frozen=True, slots=True)
class StrategicAssessment:
    own_drones: int
    own_buildings: int
    enemy_tracks: int          # non-decoy tracks
    resource_nodes: int
    under_attack: bool
    economy_ready: bool
    posture: Posture
    score_lead: int


# --- pure assessment ---

def _min_distance_to_base(coord: Coord, base_tiles: Sequence[Coord]) -> int | None:
    if not base_tiles:
        return None
    return min(hexmath.hex_distance_cube(coord[0], coord[1], b[0], b[1]) for b in base_tiles)


def assess(context, score: ScoreState, config: StrategistConfig) -> StrategicAssessment:
    """Read the planning context into a doctrine-ready assessment. Pure w.r.t.
    the context (no mutation)."""
    world = context.world
    tracks = context.tracks
    now = world.cycle

    drones = list(world.drones())
    buildings = list(world.buildings())
    base_tiles: list[Coord] = []
    for b in buildings:
        if b.tiles:
            base_tiles.extend(b.tiles)
        elif b.origin is not None:
            base_tiles.append(b.origin)

    live_tracks = [t for t in tracks.tracks() if not t.is_decoy]

    # Under attack: any live enemy track within the base-defense radius, or a
    # non-zero threat-map cost on one of our own building tiles.
    under_attack = False
    for t in live_tracks:
        d = _min_distance_to_base(t.coord, base_tiles)
        if d is not None and d <= config.base_defense_radius:
            under_attack = True
            break
    if not under_attack and base_tiles:
        # One snapshot for all base tiles — cost_at() rebuilds the threat
        # snapshot per call, which is O(sources) per tile.
        costs = context.threat_map.costs_at(base_tiles, now_cycle=now)
        if any(cost > 0.0 for cost in costs.values()):
            under_attack = True

    lead = score.lead
    if lead >= config.protect_margin:
        posture = Posture.PROTECT
    elif lead <= -config.aggro_margin:
        posture = Posture.AGGRESSIVE
    else:
        posture = Posture.NEUTRAL

    has_production = any(
        (b.building_type or "").endswith("production_plant")
        or (b.building_type or "") == "command_center"
        for b in buildings
    )
    economy_ready = len(drones) >= config.min_economy_drones and has_production

    return StrategicAssessment(
        own_drones=len(drones),
        own_buildings=len(buildings),
        enemy_tracks=len(live_tracks),
        resource_nodes=sum(1 for _ in world.resource_nodes()),
        under_attack=under_attack,
        economy_ready=economy_ready,
        posture=posture,
        score_lead=lead,
    )


# --- pure doctrine FSM ---

def next_doctrine(current: Doctrine, a: StrategicAssessment) -> Doctrine:
    """Transition rules (highest precedence first):

    1. Enemy near base  → DEFEND (always).
    2. Behind on score (AGGRESSIVE) → ALL_IN once the economy can sustain it,
       else BUILD_UP (you cannot force trades with no drones).
    3. Ahead on score (PROTECT) → grow safely: EXPAND if ready, else BUILD_UP;
       never RAID/ALL_IN while protecting a lead.
    4. Even (NEUTRAL) → RAID when the economy is ready and enemies are known,
       else EXPAND / BUILD_UP.
    """
    if a.under_attack:
        return Doctrine.DEFEND
    if a.posture is Posture.AGGRESSIVE:
        return Doctrine.ALL_IN if a.economy_ready else Doctrine.BUILD_UP
    if a.posture is Posture.PROTECT:
        return Doctrine.EXPAND if a.economy_ready else Doctrine.BUILD_UP
    # NEUTRAL
    if a.economy_ready and a.enemy_tracks > 0:
        return Doctrine.RAID
    return Doctrine.EXPAND if a.economy_ready else Doctrine.BUILD_UP


# --- pure task generation ---

def _command_center_id(world) -> str | None:
    for b in world.buildings():
        if (b.building_type or "") == "command_center":
            return b.building_id
    # Fall back to any building as a drop-off if the CC isn't identified yet.
    for b in world.buildings():
        return b.building_id
    return None


def _next_research_key(context, config: StrategistConfig) -> str | None:
    """First research item in the plan we don't yet hold, if the economy layer
    exposes completed research; otherwise the first item (base controller is
    idempotent about already-known research)."""
    completed = getattr(context, "research_completed", None)
    for key in config.research_order:
        if completed is None or key not in completed:
            return key
    return None


def generate_tasks(
    doctrine: Doctrine, context, assessment: StrategicAssessment, config: StrategistConfig
) -> list[Task]:
    """Concrete task set for this planning round. Economy tasks form the
    backbone in every non-emergency doctrine; combat tasks layer on top."""
    world = context.world
    tasks: list[Task] = []

    dropoff = _command_center_id(world)

    # Economy backbone: mine every known resource node (the allocator caps how
    # many drones actually take each).
    node_count = 0
    if dropoff is not None and doctrine is not Doctrine.ALL_IN:
        for i, node in enumerate(world.resource_nodes()):
            node_count += 1
            tasks.append(
                MineLoop(
                    task_id=f"mine:{node.q},{node.r}",
                    resource_node=(node.q, node.r),
                    dropoff_id=dropoff,
                    resource_type=node.resource_type,
                    priority=1.0 if doctrine is Doctrine.DEFEND else 1.5,
                )
            )

    # No known ore yet -> bootstrap scouting so the fleet isn't idle. Centered
    # on the CC; dissolves naturally once a node is found (mine tasks appear
    # and the changed assignment replaces the scout plan).
    if node_count == 0 and doctrine in (Doctrine.BUILD_UP, Doctrine.EXPAND):
        center = (0, 0)
        cc = world.get_building(dropoff) if dropoff is not None else None
        if cc is not None and cc.origin is not None:
            center = cc.origin
        tasks.append(
            ScoutSector(
                task_id="scout:bootstrap",
                center=center,
                radius=8,
                max_assignees=2,
            )
        )

    # Research + production (building work; base controller executes).
    research_key = _next_research_key(context, config)
    if research_key is not None and doctrine in (Doctrine.BUILD_UP, Doctrine.EXPAND):
        tasks.append(Research(task_id=f"research:{research_key}", research_key=research_key))
    if assessment.own_drones < config.drone_target and doctrine in (
        Doctrine.BUILD_UP,
        Doctrine.EXPAND,
    ):
        # The fleet size is part of the id so the task reads as *changed* after
        # a drone is gained or lost — a one-shot production plan re-installs.
        tasks.append(
            ProduceDrone(
                task_id=f"produce:miner:{assessment.own_drones}",
                equipment=("drill", "hopper", "sensors"),
            )
        )

    # Combat: strike known enemies when pressing or defending.
    if doctrine in (Doctrine.DEFEND, Doctrine.RAID, Doctrine.ALL_IN):
        for t in context.tracks.tracks():
            if t.is_decoy:
                continue
            tasks.append(
                Strike(
                    task_id=f"strike:{t.track_id}",
                    target=t.drone_id or (t.q, t.r),
                    priority=2.0 if doctrine is Doctrine.DEFEND else 1.5,
                    max_assignees=2,
                )
            )

    return tasks


# --- default fitness ---

# Task kinds a drone can be allocated to vs. work that runs on a building.
DRONE_TASK_KINDS = frozenset(
    {"mine_loop", "strike", "escort", "scout_sector", "lay_mines", "deploy_repeater"}
)
BUILDING_TASK_KINDS = frozenset({"research", "produce_drone"})


def default_fitness(entity, task: Task) -> float | None:
    """Score a (drone, task) pairing. Closer drones score higher for positional
    tasks; non-positional drone tasks fall back to the task priority. Building
    work (research/production) returns ``None`` — a drone can't do it."""
    if task.kind in BUILDING_TASK_KINDS:
        return None
    coord = getattr(entity, "coord", None)
    base = float(task.priority)
    if isinstance(task, MineLoop):
        if coord is None:
            return base * 0.5  # can still be tasked once positioned
        d = hexmath.hex_distance_cube(coord[0], coord[1], task.resource_node[0], task.resource_node[1])
        return base / (1.0 + d)
    if isinstance(task, Strike) and not isinstance(task.target, str):
        if coord is None:
            return base * 0.5
        d = hexmath.hex_distance_cube(coord[0], coord[1], task.target[0], task.target[1])
        return base / (1.0 + d)
    return base


# --- controller wiring ---

# A factory builds the pipeliner ControllerOutput that executes ``task`` on
# ``entity`` in ``context``. Keyed by task ``kind``. Injected by the runtime.
ControllerFactory = Callable[[object, Task, object], ControllerOutput]


@dataclass(slots=True)
class Strategist:
    """Installable :class:`~agent.runtime.PlanningStrategy`.

    ``score_source`` is an async callable returning the latest :class:`ScoreState`
    (the runtime wires it to scoreboard polling). ``controller_factories`` maps a
    task ``kind`` to a factory that produces the pipeliner output for an
    assignment; unmapped kinds are allocated but not executed yet (logged).
    """

    score_source: Callable[[], "asyncio.Future[ScoreState] | ScoreState"] | None = None
    controller_factories: Mapping[str, ControllerFactory] = field(default_factory=dict)
    config: StrategistConfig = field(default_factory=StrategistConfig)
    fitness: Callable[[object, Task], float | None] = default_fitness
    doctrine: Doctrine = Doctrine.BUILD_UP
    _context: object | None = field(default=None, init=False, repr=False)
    _task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _stop: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    # (entity kind, entity id) -> task_id currently installed in the pipeliner.
    _installed: dict[tuple[str, str], str] = field(default_factory=dict, init=False, repr=False)

    # PlanningStrategy protocol
    async def start(self, context) -> None:
        self._context = context
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="strategist")
        await asyncio.sleep(0)

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await self._task
            finally:
                self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # a planning glitch must never kill the loop
                log.exception("strategist tick failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.config.tick_interval_s)
            except asyncio.TimeoutError:
                pass

    async def _read_score(self) -> ScoreState:
        if self.score_source is None:
            return ScoreState()
        result = self.score_source()
        if asyncio.isfuture(result) or asyncio.iscoroutine(result):
            result = await result
        return result if isinstance(result, ScoreState) else ScoreState()

    async def tick(self) -> Doctrine:
        """One planning round. Returns the doctrine chosen (handy for tests)."""
        assert self._context is not None, "start() must run before tick()"
        context = self._context

        score = await self._read_score()
        assessment = assess(context, score, self.config)
        doctrine = next_doctrine(self.doctrine, assessment)
        if doctrine != self.doctrine:
            log.info("doctrine %s -> %s (%s)", self.doctrine.value, doctrine.value, assessment)
        self.doctrine = doctrine

        tasks = generate_tasks(doctrine, context, assessment, self.config)
        drone_tasks = [t for t in tasks if t.kind in DRONE_TASK_KINDS]
        building_tasks = [t for t in tasks if t.kind in BUILDING_TASK_KINDS]

        drones = list(context.world.drones())
        if drones and drone_tasks:
            result = context.allocator.allocate(drones, drone_tasks, self.fitness)
            for entity_id, task in result.items():
                await self._register("drone", entity_id, result.assignment_for(entity_id).entity, task, context)

        for task in building_tasks:
            building = self._target_building(context.world, task)
            if building is not None:
                await self._register("building", building.building_id, building, task, context)

        self._prune_installed(context)
        return doctrine

    async def _register(self, kind: str, entity_id: str, entity, task: Task, context) -> None:
        """Install an entity's controller output in the pipeliner, but only on
        assignment *change*. An unchanged task is left alone — its already-
        installed factory keeps replanning from live state, and replacing it
        every tick would flush the server queue and defeat queue saturation.
        A fresh entity is ``register``ed; a changed one is switched with the
        async ``replace_plan`` (flush + rebuild)."""
        key = (kind, entity_id)
        if self._installed.get(key) == task.task_id:
            return
        factory = self.controller_factories.get(task.kind)
        if factory is None:
            log.debug("no controller factory for task kind %s; allocated but not executed", task.kind)
            return
        output = factory(entity, task, context)
        if key in self._installed:
            await context.pipeliner.replace_plan(kind, entity_id, output)
        else:
            context.pipeliner.register(kind, entity_id, output)
        self._installed[key] = task.task_id

    def _prune_installed(self, context) -> None:
        """Forget installations for entities that no longer exist (destroyed
        drones / lost buildings) so the bookkeeping can't grow without bound
        and a re-created id starts from a clean register."""
        live = {("drone", d.drone_id) for d in context.world.drones()}
        live |= {("building", b.building_id) for b in context.world.buildings()}
        for key in [k for k in self._installed if k not in live]:
            del self._installed[key]

    @staticmethod
    def _target_building(world, task: Task):
        """Pick the building that should run a building task: the drone plant for
        production, else the command center; the command center for research."""
        buildings = list(world.buildings())
        if task.kind == "produce_drone":
            for b in buildings:
                if (b.building_type or "").endswith("drone_production_plant"):
                    return b
        for b in buildings:
            if (b.building_type or "") == "command_center":
                return b
        return buildings[0] if buildings else None
