"""Commander HTTP snapshot and Socket.IO update stream.

The commander facade projects the agent's internal dataclasses onto the
versioned DTOs in :mod:`agent.commander.contract`.  It subscribes to the world,
track, and planning stores instead of polling them, which gives the UI current
``GET /v1/state`` and ``GET /v1/plan`` snapshots plus sequence-numbered updates.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import socketio
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from agent.bus import EventBus
from agent.commander.contract import (
    CONTRACT_VERSION,
    Alert,
    BuildingState,
    BuildingUpsert,
    Coordinate,
    Directive,
    DroneState,
    DroneUpsert,
    EnemyTrackState,
    EnemyTrackUpsert,
    EntityRemoved,
    PlanSnapshot,
    StateSnapshot,
    TileState,
    TileRemoved,
    TileUpsert,
    WorldChange as ContractWorldChange,
    WorldDiff,
)
from agent.commander.directives import DirectiveStore
from agent.world.model import TOPIC_WORLD_CHANGED, WorldChange, WorldModel
from agent.world.tiles import TileBelief
from agent.world.tracks import TOPIC_TRACK_CHANGED, EnemyTrack, TrackStore


TOPIC_PLAN_CHANGED = "planning.changed"
TOPIC_COMMANDER_ALERT = "commander.alert"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _coordinate(coord: tuple[int, int] | None) -> Coordinate | None:
    if coord is None:
        return None
    return Coordinate(q=coord[0], r=coord[1])


def _tile_state(tile: TileBelief, cycle: int) -> TileState:
    return TileState(
        coordinate=Coordinate(q=tile.q, r=tile.r),
        terrain_type=tile.terrain_type,
        elevation=tile.elevation,
        has_resource=tile.has_resource,
        resource_type=tile.resource_type,
        resource_amount=tile.resource_amount,
        has_building=tile.has_building,
        building_id=tile.building_id,
        has_drone=tile.has_drone,
        last_seen=tile.last_seen,
        source=tile.source.name.lower(),
        confidence=tile.confidence(cycle),
    )


def _drone_state(record: Any) -> DroneState:
    return DroneState(
        drone_id=record.drone_id,
        coordinate=_coordinate(record.coord),
        direction=record.direction,
        elevation=record.elevation,
        last_seen=record.last_seen,
    )


def _building_state(record: Any, allegiance: str) -> BuildingState:
    return BuildingState(
        building_id=record.building_id,
        building_type=record.building_type,
        allegiance=allegiance,
        origin=_coordinate(record.origin),
        tiles=[Coordinate(q=q, r=r) for q, r in record.tiles],
        last_seen=record.last_seen,
    )


def _track_state(track: EnemyTrack, cycle: int) -> EnemyTrackState:
    predicted = track.predict(cycle)
    return EnemyTrackState(
        track_id=track.track_id,
        drone_id=track.drone_id,
        coordinate=Coordinate(q=track.q, r=track.r),
        predicted_coordinate=Coordinate(q=predicted[0], r=predicted[1]),
        first_seen=track.first_seen,
        last_seen=track.last_seen,
        source=track.source.name.lower(),
        confidence=track.confidence(cycle),
        is_decoy=track.is_decoy,
        equipment=sorted(track.equipment),
        equipment_known=track.equipment_known,
    )


class CommanderAPI:
    """Own the FastAPI application, Socket.IO server, and state projection.

    Constructing the facade attaches its bus consumers immediately.  Call
    :meth:`close` during shutdown to detach them.
    """

    def __init__(
        self,
        world: WorldModel,
        tracks: TrackStore,
        bus: EventBus,
        *,
        directives: DirectiveStore | None = None,
        socket_server: socketio.AsyncServer | None = None,
        cors_allowed_origins: str | list[str] = "*",
    ) -> None:
        self.world = world
        self.tracks = tracks
        self.bus = bus
        self.directives = directives or DirectiveStore(bus)
        self._owns_directives = directives is None
        self.sio = socket_server or socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins=cors_allowed_origins,
        )

        @asynccontextmanager
        async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
            try:
                yield
            finally:
                await self.close()

        self.http = FastAPI(
            title="Drone CTF Commander API",
            version=CONTRACT_VERSION,
            lifespan=lifespan,
        )
        # The Socket.IO server has its own allow-list; the HTTP snapshot routes
        # need the matching one so a browser control-plane can seed from them.
        self.cors_allowed_origins = (
            [cors_allowed_origins]
            if isinstance(cors_allowed_origins, str)
            else list(cors_allowed_origins)
        )
        self.http.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_allowed_origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
        self.app = socketio.ASGIApp(self.sio, other_asgi_app=self.http)
        self._lock = asyncio.Lock()
        self._sequence = 0
        self._closed = False

        self._tiles: dict[tuple[int, int], TileState] = {}
        self._drones: dict[str, DroneState] = {}
        self._buildings: dict[str, BuildingState] = {}
        self._tracks: dict[str, EnemyTrackState] = {}
        self._cycle = world.cycle
        self._plan = PlanSnapshot(
            contract_version=CONTRACT_VERSION,
            sequence=0,
            cycle=self._cycle,
            tasks=[],
            assignments=[],
            paths=[],
        )
        self._refresh_all()

        self._unsubscribers = [
            bus.subscribe(TOPIC_WORLD_CHANGED, self._on_world_change),
            bus.subscribe(TOPIC_TRACK_CHANGED, self._on_track_change),
            bus.subscribe(TOPIC_PLAN_CHANGED, self._on_plan_changed),
            bus.subscribe(TOPIC_COMMANDER_ALERT, self._on_alert),
        ]

        self.http.add_api_route(
            "/v1/state",
            self.state,
            methods=["GET"],
            response_model=StateSnapshot,
        )
        self.http.add_api_route(
            "/v1/directives",
            self.submit_directive,
            methods=["POST"],
            response_model=Directive,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.http.add_api_route(
            "/v1/plan",
            self.plan,
            methods=["GET"],
            response_model=PlanSnapshot,
        )

    async def state(self) -> StateSnapshot:
        """Return an atomic snapshot aligned with the diff sequence."""

        async with self._lock:
            return StateSnapshot(
                contract_version=CONTRACT_VERSION,
                sequence=self._sequence,
                cycle=self._cycle,
                generated_at=_now(),
                tiles=[self._tiles[key] for key in sorted(self._tiles)],
                drones=[self._drones[key] for key in sorted(self._drones)],
                buildings=[self._buildings[key] for key in sorted(self._buildings)],
                enemy_tracks=[self._tracks[key] for key in sorted(self._tracks)],
            )

    async def plan(self) -> PlanSnapshot:
        """Return the latest complete planning snapshot."""

        async with self._lock:
            return self._plan.model_copy(deep=True)

    async def publish_plan(self, plan: PlanSnapshot | Mapping[str, Any]) -> None:
        """Retain and push a validated planning snapshot to commander clients."""

        snapshot = (
            plan.model_copy(deep=True)
            if isinstance(plan, PlanSnapshot)
            else PlanSnapshot.model_validate(plan)
        )
        async with self._lock:
            self._plan = snapshot
        await self._emit("plan_changed", snapshot)

    async def submit_directive(self, directive: Directive) -> Directive:
        """Validate and activate a stance, squad order, or entity override."""

        await self.directives.apply(directive)
        return directive

    async def publish_alert(self, alert: Alert | Mapping[str, Any]) -> None:
        """Push a validated alert to all commander clients."""

        payload = alert if isinstance(alert, Alert) else Alert.model_validate(alert)
        await self._emit("alert", payload)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_directives:
            await self.directives.close()
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()

    def _refresh_all(self) -> None:
        """Refresh the projection synchronously on the single event loop."""

        cycle = self.world.cycle
        self._cycle = cycle
        self._tiles = {tile.coord: _tile_state(tile, cycle) for tile in self.world.tiles()}
        self._drones = {
            record.drone_id: _drone_state(record) for record in self.world.drones()
        }
        buildings = {
            record.building_id: _building_state(record, "friendly")
            for record in self.world.buildings()
        }
        buildings.update(
            {
                record.building_id: _building_state(record, "enemy")
                for record in self.world.enemy_buildings()
            }
        )
        self._buildings = buildings
        self._tracks = {
            track.track_id: _track_state(track, cycle) for track in self.tracks.tracks()
        }

    async def _on_world_change(self, _topic: str, change: WorldChange) -> None:
        alert: Alert | None = None
        async with self._lock:
            changes = self._project_world_change(change)
            if change.kind == "enemy_building" and change.keys:
                building = self.world.get_enemy_building(str(change.keys[0]))
                if building is not None:
                    alert = self._threat_alert(
                        code="enemy_building_detected",
                        message=f"Enemy building detected: {building.building_id}",
                        entity_ids=[building.building_id],
                        coordinate=_coordinate(building.origin),
                        critical=building.building_type is not None,
                    )
            diff = self._next_diff(changes, cycle=change.cycle)
        if diff is not None:
            await self._emit("world_diff", diff)
        if alert is not None:
            await self._emit("alert", alert)

    def _project_world_change(self, change: WorldChange) -> list[ContractWorldChange]:
        self._cycle = self.world.cycle
        if change.kind == "restored":
            old_tiles = set(self._tiles)
            old_drones = set(self._drones)
            old_buildings = set(self._buildings)
            self._refresh_all()
            changes: list[ContractWorldChange] = [
                EntityRemoved(type="entity_removed", entity_type="drone", entity_id=entity_id)
                for entity_id in sorted(old_drones - self._drones.keys())
            ]
            changes.extend(
                EntityRemoved(
                    type="entity_removed", entity_type="building", entity_id=entity_id
                )
                for entity_id in sorted(old_buildings - self._buildings.keys())
            )
            changes.extend(
                TileRemoved(type="tile_removed", coordinate=Coordinate(q=q, r=r))
                for q, r in sorted(old_tiles - self._tiles.keys())
            )
            changes.extend(
                TileUpsert(type="tile_upsert", tile=value)
                for value in self._tiles.values()
            )
            changes.extend(
                DroneUpsert(type="drone_upsert", drone=value) for value in self._drones.values()
            )
            changes.extend(
                BuildingUpsert(type="building_upsert", building=value)
                for value in self._buildings.values()
            )
            return changes

        changes = []
        if change.kind in {"tile", "resource"}:
            for key in change.keys:
                if not isinstance(key, tuple) or len(key) != 2:
                    continue
                tile = self.world.get_tile(key)
                if tile is not None:
                    state = _tile_state(tile, self._cycle)
                    self._tiles[key] = state
                    changes.append(TileUpsert(type="tile_upsert", tile=state))
        elif change.kind == "drone":
            for key in change.keys:
                record = self.world.get_drone(str(key))
                if record is not None:
                    state = _drone_state(record)
                    self._drones[record.drone_id] = state
                    changes.append(DroneUpsert(type="drone_upsert", drone=state))
        elif change.kind in {"building", "enemy_building"}:
            enemy = change.kind == "enemy_building"
            for key in change.keys:
                record = (
                    self.world.get_enemy_building(str(key))
                    if enemy
                    else self.world.get_building(str(key))
                )
                if record is not None:
                    state = _building_state(record, "enemy" if enemy else "friendly")
                    self._buildings[record.building_id] = state
                    changes.append(BuildingUpsert(type="building_upsert", building=state))
        elif change.kind == "removed":
            for key in change.keys:
                entity_id = str(key)
                if entity_id in self._drones and self.world.get_drone(entity_id) is None:
                    self._drones.pop(entity_id)
                    changes.append(
                        EntityRemoved(
                            type="entity_removed", entity_type="drone", entity_id=entity_id
                        )
                    )
                if entity_id in self._buildings and (
                    self.world.get_building(entity_id) is None
                    and self.world.get_enemy_building(entity_id) is None
                ):
                    self._buildings.pop(entity_id)
                    changes.append(
                        EntityRemoved(
                            type="entity_removed", entity_type="building", entity_id=entity_id
                        )
                    )
        return changes

    async def _on_track_change(self, _topic: str, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            return
        track_id = payload.get("track_id")
        if track_id is None:
            bearings = self.tracks.recent_bearings()
            if bearings:
                bearing = bearings[-1]
                alert = Alert(
                    alert_id=str(uuid4()),
                    severity="warning",
                    code="enemy_bearing_detected",
                    message=f"Enemy bearing detected in direction {bearing.direction}",
                    cycle=bearing.cycle,
                    created_at=_now(),
                    entity_ids=[],
                    coordinate=Coordinate(q=bearing.sensor_q, r=bearing.sensor_r),
                )
                await self._emit("alert", alert)
            return
        if not isinstance(track_id, str):
            return

        alert: Alert | None = None
        async with self._lock:
            self._cycle = self.world.cycle
            if payload.get("removed"):
                if self._tracks.pop(track_id, None) is None:
                    return
                change: ContractWorldChange = EntityRemoved(
                    type="entity_removed", entity_type="enemy_track", entity_id=track_id
                )
            else:
                track = self.tracks.get(track_id)
                if track is None:
                    return
                state = _track_state(track, self._cycle)
                self._tracks[track_id] = state
                change = EnemyTrackUpsert(type="enemy_track_upsert", enemy_track=state)
                if not track.is_decoy:
                    alert = self._threat_alert(
                        code="enemy_track_detected",
                        message=f"Enemy track detected: {track_id}",
                        entity_ids=[track.drone_id or track_id],
                        coordinate=state.coordinate,
                        critical=bool(track.equipment),
                    )
            diff = self._next_diff([change], cycle=self._cycle)
        if diff is not None:
            await self._emit("world_diff", diff)
        if alert is not None:
            await self._emit("alert", alert)

    async def _on_plan_changed(self, _topic: str, payload: Any) -> None:
        await self.publish_plan(payload)

    async def _on_alert(self, _topic: str, payload: Any) -> None:
        await self.publish_alert(payload)

    def _next_diff(
        self, changes: list[ContractWorldChange], *, cycle: int
    ) -> WorldDiff | None:
        if not changes:
            return None
        self._cycle = max(self._cycle, cycle)
        self._sequence += 1
        return WorldDiff(
            contract_version=CONTRACT_VERSION,
            sequence=self._sequence,
            cycle=self._cycle,
            changes=changes,
        )

    def _threat_alert(
        self,
        *,
        code: str,
        message: str,
        entity_ids: list[str],
        coordinate: Coordinate | None,
        critical: bool,
    ) -> Alert:
        return Alert(
            alert_id=str(uuid4()),
            severity="critical" if critical else "warning",
            code=code,
            message=message,
            cycle=self._cycle,
            created_at=_now(),
            entity_ids=entity_ids,
            coordinate=coordinate,
        )

    async def _emit(self, event: str, payload: Any) -> None:
        await self.sio.emit(event, payload.model_dump(mode="json"))


def create_app(
    world: WorldModel,
    tracks: TrackStore,
    bus: EventBus,
    **kwargs: Any,
) -> socketio.ASGIApp:
    """Create the combined Socket.IO/FastAPI ASGI application.

    The owning :class:`CommanderAPI` is available as ``app.commander`` for
    lifecycle management and tests.
    """

    commander = CommanderAPI(world, tracks, bus, **kwargs)
    app = commander.app
    app.commander = commander  # type: ignore[attr-defined]
    return app


__all__ = [
    "CommanderAPI",
    "TOPIC_COMMANDER_ALERT",
    "TOPIC_PLAN_CHANGED",
    "create_app",
]
