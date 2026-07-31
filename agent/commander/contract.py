"""Versioned DTOs shared by the commander API and control-plane.

These Pydantic models are the runtime validation source of truth.  The
hand-written TypeScript package in ``packages/contract`` mirrors them, while
``tools/contract_schema`` generates JSON Schema and compile-time compatibility
types so CI can detect drift in either direction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, RootModel


CONTRACT_VERSION = "1.0"


class ContractModel(BaseModel):
    """Strict base class for externally visible commander payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Coordinate(ContractModel):
    q: int
    r: int


class TileState(ContractModel):
    coordinate: Coordinate
    terrain_type: int | None
    elevation: int | None
    has_resource: bool
    resource_type: str | None
    resource_amount: int | None
    has_building: bool
    building_id: str | None
    has_drone: bool
    last_seen: int = Field(ge=0)
    source: Literal["scan", "identify", "status_report", "admin"]
    confidence: float = Field(ge=0.0, le=1.0)


class DroneState(ContractModel):
    drone_id: str
    coordinate: Coordinate | None
    direction: int | None = Field(ge=0, le=5)
    elevation: int
    last_seen: int = Field(ge=0)


class BuildingState(ContractModel):
    building_id: str
    building_type: str | None
    allegiance: Literal["friendly", "enemy"]
    origin: Coordinate | None
    tiles: list[Coordinate]
    last_seen: int = Field(ge=0)


class EnemyTrackState(ContractModel):
    track_id: str
    drone_id: str | None
    coordinate: Coordinate
    predicted_coordinate: Coordinate
    first_seen: int = Field(ge=0)
    last_seen: int = Field(ge=0)
    source: Literal["scan", "identify", "seismic", "df_antenna"]
    confidence: float = Field(ge=0.0, le=1.0)
    is_decoy: bool
    equipment: list[str]
    equipment_known: bool


class StateSnapshot(ContractModel):
    contract_version: Literal["1.0"]
    sequence: int = Field(ge=0)
    cycle: int = Field(ge=0)
    generated_at: datetime
    tiles: list[TileState]
    drones: list[DroneState]
    buildings: list[BuildingState]
    enemy_tracks: list[EnemyTrackState]


class TileUpsert(ContractModel):
    type: Literal["tile_upsert"]
    tile: TileState


class DroneUpsert(ContractModel):
    type: Literal["drone_upsert"]
    drone: DroneState


class BuildingUpsert(ContractModel):
    type: Literal["building_upsert"]
    building: BuildingState


class EnemyTrackUpsert(ContractModel):
    type: Literal["enemy_track_upsert"]
    enemy_track: EnemyTrackState


class TileRemoved(ContractModel):
    type: Literal["tile_removed"]
    coordinate: Coordinate


class EntityRemoved(ContractModel):
    type: Literal["entity_removed"]
    entity_type: Literal["drone", "building", "enemy_track"]
    entity_id: str


WorldChange: TypeAlias = Annotated[
    TileUpsert
    | DroneUpsert
    | BuildingUpsert
    | EnemyTrackUpsert
    | TileRemoved
    | EntityRemoved,
    Field(discriminator="type"),
]


class WorldDiff(ContractModel):
    contract_version: Literal["1.0"]
    sequence: int = Field(ge=1)
    cycle: int = Field(ge=0)
    changes: list[WorldChange]


Scalar: TypeAlias = str | int | float | bool | None


class PlanTask(ContractModel):
    task_id: str
    kind: str
    priority: float = Field(gt=0.0)
    max_assignees: int = Field(ge=1)
    parameters: dict[str, Scalar]


class PlanAssignment(ContractModel):
    entity_id: str
    task_id: str
    fitness: float | None
    pinned: bool
    retained: bool


class PlannedPath(ContractModel):
    entity_id: str
    coordinates: list[Coordinate]


class PlanSnapshot(ContractModel):
    contract_version: Literal["1.0"]
    sequence: int = Field(ge=0)
    cycle: int = Field(ge=0)
    tasks: list[PlanTask]
    assignments: list[PlanAssignment]
    paths: list[PlannedPath]


class DirectiveFields(ContractModel):
    directive_id: str
    issued_at: datetime
    ttl_seconds: int = Field(gt=0)


class StanceDirective(DirectiveFields):
    kind: Literal["stance"]
    stance: Literal["aggressive", "balanced", "defensive", "retreat"]


class SquadOrderDirective(DirectiveFields):
    kind: Literal["squad_order"]
    squad_id: str
    order: Literal[
        "move",
        "hold",
        "mine",
        "scout",
        "escort",
        "strike",
        "lay_mines",
        "deploy_repeater",
    ]
    target: Coordinate | str | None


class EntityOverrideDirective(DirectiveFields):
    kind: Literal["entity_override"]
    entity_id: str
    task_id: str


Directive: TypeAlias = Annotated[
    StanceDirective | SquadOrderDirective | EntityOverrideDirective,
    Field(discriminator="kind"),
]


class Alert(ContractModel):
    alert_id: str
    severity: Literal["info", "warning", "critical"]
    code: str
    message: str
    cycle: int = Field(ge=0)
    created_at: datetime
    entity_ids: list[str]
    coordinate: Coordinate | None


class WorldDiffEvent(ContractModel):
    event: Literal["world_diff"]
    data: WorldDiff


class PlanChangedEvent(ContractModel):
    event: Literal["plan_changed"]
    data: PlanSnapshot


class AlertEvent(ContractModel):
    event: Literal["alert"]
    data: Alert


SocketEvent: TypeAlias = Annotated[
    WorldDiffEvent | PlanChangedEvent | AlertEvent,
    Field(discriminator="event"),
]


ContractDocument: TypeAlias = (
    StateSnapshot | WorldDiff | PlanSnapshot | Directive | SocketEvent
)


class ContractV1(RootModel[ContractDocument]):
    """JSON-Schema root that includes every public v1 contract definition."""


__all__ = [
    "CONTRACT_VERSION",
    "Alert",
    "AlertEvent",
    "BuildingState",
    "BuildingUpsert",
    "ContractDocument",
    "ContractV1",
    "Coordinate",
    "Directive",
    "DroneState",
    "DroneUpsert",
    "EnemyTrackState",
    "EnemyTrackUpsert",
    "EntityOverrideDirective",
    "EntityRemoved",
    "PlanAssignment",
    "PlanChangedEvent",
    "PlanSnapshot",
    "PlanTask",
    "PlannedPath",
    "Scalar",
    "SocketEvent",
    "SquadOrderDirective",
    "StanceDirective",
    "StateSnapshot",
    "TileRemoved",
    "TileState",
    "TileUpsert",
    "WorldChange",
    "WorldDiff",
    "WorldDiffEvent",
]
