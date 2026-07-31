"""World model: our persistent belief about the map and our own assets.

The model owns tile beliefs, our own drones and buildings, identified hostile
buildings, and the set of known resource nodes. Ingest translates parsed wire
messages into calls on this API; every mutation publishes a
:data:`TOPIC_WORLD_CHANGED` event so downstream consumers (threat map, planners,
the commander diff stream) can react without polling.

Coordinates are local-absolute odd-q offset (see :mod:`agent.rules.hexmath`);
converting relative wire coordinates to absolute is the ingest layer's job.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from agent.bus import EventBus
from agent.world.tiles import (
    DEFAULT_CONFIDENCE_HALF_LIFE,
    Coord,
    Source,
    TileBelief,
)

TOPIC_WORLD_CHANGED = "world.changed"


@dataclass(slots=True)
class DroneRecord:
    """A drone we own (or a friendly). Position/heading only — battery/ammo are
    tracked by the entity sim (E2.4); enemy drones live in tracks (E3.3)."""

    drone_id: str
    q: int | None = None
    r: int | None = None
    direction: int | None = None
    elevation: int = 0
    last_seen: int = 0

    @property
    def coord(self) -> Coord | None:
        if self.q is None or self.r is None:
            return None
        return (self.q, self.r)


@dataclass(slots=True)
class BuildingRecord:
    """A building we own. ``origin`` is its reference tile; ``tiles`` its full
    footprint in absolute coordinates."""

    building_id: str
    building_type: str | None = None
    origin: Coord | None = None
    tiles: tuple[Coord, ...] = ()
    last_seen: int = 0


@dataclass(slots=True)
class EnemyBuildingRecord:
    """An identified hostile building.

    Scans reveal only that a building occupies a tile; identify supplies the
    type needed to choose a weapon envelope.  Enemy buildings live separately
    from our authoritative status-report inventory so a friendly status sweep
    cannot accidentally remove them.
    """

    building_id: str
    building_type: str | None = None
    origin: Coord | None = None
    tiles: tuple[Coord, ...] = ()
    last_seen: int = 0


@dataclass(frozen=True, slots=True)
class WorldChange:
    """Describes what changed, published on the bus after a mutation.

    ``kind`` is one of: ``tile``, ``drone``, ``building``,
    ``enemy_building``, ``resource``, ``removed``, ``cycle``, ``restored``. ``keys``
    identifies the affected entities (tile coords, drone/building ids).
    ``cycle`` is the model's current cycle.
    """

    kind: str
    keys: tuple = ()
    cycle: int = 0


@dataclass(frozen=True, slots=True)
class WorldState:
    """A detached, internally consistent checkpoint of a :class:`WorldModel`.

    The records inside the tuples are defensive copies.  Capturing a state is
    synchronous on purpose: in the single-event-loop world-model architecture,
    no mutation can interleave with the capture.  Persistence can then encode
    and write this detached value in a worker thread.
    """

    cycle: int
    confidence_half_life: int
    tiles: tuple[TileBelief, ...]
    drones: tuple[DroneRecord, ...]
    buildings: tuple[BuildingRecord, ...]
    enemy_buildings: tuple[EnemyBuildingRecord, ...]


@dataclass(slots=True)
class _Observation:
    """A single tile observation handed to :meth:`WorldModel.observe_tiles`."""

    q: int
    r: int
    terrain_type: int | None = None
    elevation: int | None = None
    has_resource: bool = False
    resource_type: str | None = None
    resource_amount: int | None = None
    has_building: bool = False
    building_id: str | None = None
    has_drone: bool = False


# Public alias so ingest can construct observations without importing a private
# name.
TileObservation = _Observation


class WorldModel:
    def __init__(
        self,
        bus: EventBus | None = None,
        *,
        confidence_half_life: int = DEFAULT_CONFIDENCE_HALF_LIFE,
    ) -> None:
        self._bus = bus
        self._half_life = confidence_half_life
        self._tiles: dict[Coord, TileBelief] = {}
        self._drones: dict[str, DroneRecord] = {}
        self._buildings: dict[str, BuildingRecord] = {}
        self._enemy_buildings: dict[str, EnemyBuildingRecord] = {}
        self._resource_nodes: set[Coord] = set()
        self._cycle = 0

    # --- clock ---

    @property
    def cycle(self) -> int:
        return self._cycle

    async def set_cycle(self, cycle: int) -> None:
        """Advance the model clock (monotonic). Messages carry a cycle count in
        their timestamp; ingest calls this so confidence decay has a reference."""
        if cycle > self._cycle:
            self._cycle = cycle
            await self._emit(WorldChange("cycle", (), self._cycle))

    # --- tiles ---

    def get_tile(self, coord: Coord) -> TileBelief | None:
        return self._tiles.get(coord)

    def tiles(self) -> Iterable[TileBelief]:
        return self._tiles.values()

    @property
    def tile_count(self) -> int:
        return len(self._tiles)

    def confidence(self, coord: Coord) -> float:
        """Confidence in the belief at ``coord`` (0.0 if we've never seen it)."""
        tile = self._tiles.get(coord)
        if tile is None:
            return 0.0
        return tile.confidence(self._cycle, self._half_life)

    def snapshot_state(self) -> WorldState:
        """Return a detached checkpoint suitable for background persistence."""
        return WorldState(
            cycle=self._cycle,
            confidence_half_life=self._half_life,
            tiles=tuple(replace(tile) for _, tile in sorted(self._tiles.items())),
            drones=tuple(replace(rec) for _, rec in sorted(self._drones.items())),
            buildings=tuple(replace(rec) for _, rec in sorted(self._buildings.items())),
            enemy_buildings=tuple(
                replace(rec) for _, rec in sorted(self._enemy_buildings.items())
            ),
        )

    async def restore_state(self, state: WorldState) -> None:
        """Atomically replace all beliefs from a previously captured checkpoint.

        Resource-node membership is derived from tile beliefs rather than
        persisted separately, preventing the two representations from drifting.
        One aggregate event is emitted after the replacement; replaying every
        historical mutation at startup would be both misleading and expensive.
        """
        if state.confidence_half_life <= 0:
            raise ValueError("confidence_half_life must be positive")

        tiles = {tile.coord: replace(tile) for tile in state.tiles}
        drones = {rec.drone_id: replace(rec) for rec in state.drones}
        buildings = {rec.building_id: replace(rec) for rec in state.buildings}
        enemy_buildings = {
            rec.building_id: replace(rec) for rec in state.enemy_buildings
        }

        self._cycle = state.cycle
        self._half_life = state.confidence_half_life
        self._tiles = tiles
        self._drones = drones
        self._buildings = buildings
        self._enemy_buildings = enemy_buildings
        self._resource_nodes = {
            coord for coord, tile in tiles.items() if tile.has_resource
        }
        await self._emit(WorldChange("restored", (), self._cycle))

    async def observe_tile(
        self, obs: _Observation, *, cycle: int, source: Source = Source.SCAN
    ) -> None:
        self._apply_observation(obs, cycle, source)
        if cycle > self._cycle:
            self._cycle = cycle
        await self._emit(WorldChange("tile", ((obs.q, obs.r),), self._cycle))

    async def observe_tiles(
        self,
        observations: Iterable[_Observation],
        *,
        cycle: int,
        source: Source = Source.SCAN,
    ) -> None:
        """Apply a batch of tile observations (e.g. one scan) and emit a single
        change event naming all affected coordinates."""
        changed: list[Coord] = []
        for obs in observations:
            self._apply_observation(obs, cycle, source)
            changed.append((obs.q, obs.r))
        if not changed:
            return
        if cycle > self._cycle:
            self._cycle = cycle
        await self._emit(WorldChange("tile", tuple(changed), self._cycle))

    def _apply_observation(self, obs: _Observation, cycle: int, source: Source) -> None:
        tile = self._tiles.get((obs.q, obs.r))
        if tile is None:
            tile = TileBelief(q=obs.q, r=obs.r)
            self._tiles[(obs.q, obs.r)] = tile
        # Only overwrite fields the observation actually carries; a scan that
        # doesn't resolve a resource shouldn't erase an identify's detail.
        if obs.terrain_type is not None:
            tile.terrain_type = obs.terrain_type
        if obs.elevation is not None:
            tile.elevation = obs.elevation
        tile.has_resource = obs.has_resource
        if obs.resource_type is not None:
            tile.resource_type = obs.resource_type
        if obs.resource_amount is not None:
            tile.resource_amount = obs.resource_amount
        tile.has_building = obs.has_building
        if obs.building_id is not None:
            tile.building_id = obs.building_id
        elif not obs.has_building:
            tile.building_id = None
        tile.has_drone = obs.has_drone
        tile.last_seen = cycle
        tile.source = source

        if obs.has_resource:
            self._resource_nodes.add((obs.q, obs.r))
        else:
            self._resource_nodes.discard((obs.q, obs.r))

    # --- resource nodes ---

    def resource_nodes(self) -> Iterable[TileBelief]:
        """Tiles currently believed to hold a resource."""
        for coord in self._resource_nodes:
            tile = self._tiles.get(coord)
            if tile is not None:
                yield tile

    # --- own drones ---

    def get_drone(self, drone_id: str) -> DroneRecord | None:
        return self._drones.get(drone_id)

    def drones(self) -> Iterable[DroneRecord]:
        return self._drones.values()

    async def upsert_drone(
        self,
        drone_id: str,
        *,
        q: int | None = None,
        r: int | None = None,
        direction: int | None = None,
        elevation: int | None = None,
        cycle: int | None = None,
    ) -> None:
        rec = self._drones.get(drone_id)
        if rec is None:
            rec = DroneRecord(drone_id=drone_id)
            self._drones[drone_id] = rec
        if q is not None:
            rec.q = q
        if r is not None:
            rec.r = r
        if direction is not None:
            rec.direction = direction
        if elevation is not None:
            rec.elevation = elevation
        if cycle is not None:
            rec.last_seen = cycle
            if cycle > self._cycle:
                self._cycle = cycle
        await self._emit(WorldChange("drone", (drone_id,), self._cycle))

    async def remove_drone(self, drone_id: str) -> None:
        if self._drones.pop(drone_id, None) is not None:
            await self._emit(WorldChange("removed", (drone_id,), self._cycle))

    # --- own buildings ---

    def get_building(self, building_id: str) -> BuildingRecord | None:
        return self._buildings.get(building_id)

    def buildings(self) -> Iterable[BuildingRecord]:
        return self._buildings.values()

    async def upsert_building(
        self,
        building_id: str,
        *,
        building_type: str | None = None,
        origin: Coord | None = None,
        tiles: Iterable[Coord] | None = None,
        cycle: int | None = None,
    ) -> None:
        rec = self._buildings.get(building_id)
        if rec is None:
            rec = BuildingRecord(building_id=building_id)
            self._buildings[building_id] = rec
        if building_type is not None:
            rec.building_type = building_type
        if origin is not None:
            rec.origin = origin
        if tiles is not None:
            rec.tiles = tuple(tiles)
        if cycle is not None:
            rec.last_seen = cycle
            if cycle > self._cycle:
                self._cycle = cycle
        await self._emit(WorldChange("building", (building_id,), self._cycle))

    async def remove_building(self, building_id: str) -> None:
        if self._buildings.pop(building_id, None) is not None:
            await self._emit(WorldChange("removed", (building_id,), self._cycle))

    # --- identified enemy buildings ---

    def get_enemy_building(self, building_id: str) -> EnemyBuildingRecord | None:
        return self._enemy_buildings.get(building_id)

    def enemy_buildings(self) -> Iterable[EnemyBuildingRecord]:
        return self._enemy_buildings.values()

    async def upsert_enemy_building(
        self,
        building_id: str,
        *,
        building_type: str | None = None,
        origin: Coord | None = None,
        tiles: Iterable[Coord] | None = None,
        cycle: int | None = None,
    ) -> None:
        rec = self._enemy_buildings.get(building_id)
        if rec is None:
            rec = EnemyBuildingRecord(building_id=building_id)
            self._enemy_buildings[building_id] = rec
        if building_type is not None:
            rec.building_type = building_type
        if origin is not None:
            rec.origin = origin
        if tiles is not None:
            rec.tiles = tuple(tiles)
        if cycle is not None:
            rec.last_seen = cycle
            if cycle > self._cycle:
                self._cycle = cycle
        await self._emit(WorldChange("enemy_building", (building_id,), self._cycle))

    async def remove_enemy_building(self, building_id: str) -> None:
        if self._enemy_buildings.pop(building_id, None) is not None:
            await self._emit(WorldChange("removed", (building_id,), self._cycle))

    async def remove_enemy_buildings_at(self, coord: Coord) -> tuple[str, ...]:
        """Forget hostile buildings whose known footprint includes ``coord``."""
        removed = tuple(
            building_id
            for building_id, rec in self._enemy_buildings.items()
            if coord == rec.origin or coord in rec.tiles
        )
        for building_id in removed:
            self._enemy_buildings.pop(building_id, None)
        if removed:
            await self._emit(WorldChange("removed", removed, self._cycle))
        return removed

    # --- emit ---

    async def _emit(self, change: WorldChange) -> None:
        if self._bus is not None:
            await self._bus.publish(TOPIC_WORLD_CHANGED, change)
