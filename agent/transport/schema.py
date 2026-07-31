"""Pure Pydantic models for messages received from the game server.

This module deliberately contains only data definitions.  Decoding, logging,
and sample recording belong at the transport boundary (see ``parsing.py``).
Models allow additional fields because the server evolves independently of the
agent; observed fields remain typed while newly-added fields are preserved.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, PrivateAttr

HexCoordinate = tuple[int, int]


class WireModel(BaseModel):
    """Base for forward-compatible wire objects."""

    model_config = ConfigDict(extra="allow")


class ResourceInventory(WireModel):
    titanium_parts: int = 0
    battery_materials: int = 0
    enriched_unobtanium: int = 0


class EquipmentStatus(WireModel):
    equipment_id: str
    equipment_type: str
    name: str
    enabled_actions: list[str]
    weight: int
    durability: int
    max_durability: int
    destroyed: bool
    is_functional: bool
    current_level: int | None = None
    efficiency: float | None = None
    base_scan_range: int | None = None
    current_identify_level: int | None = None
    capacity: int | None = None
    current_load: ResourceInventory | None = None
    upgrade_tier_applied: int | None = None


class DroneStatus(WireModel):
    drone_id: str
    elevation: int
    direction: int
    speed: int | float
    max_battery: int
    current_battery: int
    max_health: int
    current_health: int
    armour: int
    equipment_slots: int
    equipment: list[EquipmentStatus]
    cargo: ResourceInventory
    shields: int
    max_shields: int
    base_shields: int
    upgrade_tier_applied: int
    chassis_weight: int
    last_action_cycle: int
    is_decoy: bool
    is_lucas: bool
    health_percentage: float
    available_equipment_slots: int


class ConstructionQueueItem(WireModel):
    """Forward-compatible construction entry; no populated entry is observed yet."""


class BuildingStatus(WireModel):
    building_id: str
    building_type: str
    owner_id: str
    size: HexCoordinate
    max_health: int
    current_health: int
    armour: int
    max_shields: int
    shields: int
    power_generation: float
    target_efficiency: float
    enabled_actions: list[str]
    under_construction: bool
    coordinates: list[HexCoordinate]
    stored_resources: ResourceInventory
    construction_queue: list[ConstructionQueueItem]
    current_construction: ConstructionQueueItem | None
    construction_progress: int | float
    resource_capacity: int
    available_capacity: int
    buildable_buildings: list[str]
    build_range: int
    unload_range: int


class IdentifiedResource(WireModel):
    """Forward-compatible identified resource; the current sample contains null."""


class IdentifiedDrone(WireModel):
    """Forward-compatible identified drone; the current sample contains an empty list."""


class IdentifiedBuilding(WireModel):
    """Forward-compatible identified building; the current sample contains null."""


class IdentifyData(WireModel):
    target_q: int
    target_r: int
    terrain: int
    elevation: int
    resource: IdentifiedResource | None
    drones: list[IdentifiedDrone]
    building: IdentifiedBuilding | None


class ScanTile(WireModel):
    coordinates: HexCoordinate
    distance: int
    terrain: str
    terrain_type: int
    elevation: int
    has_resource: bool
    has_drone: bool
    has_building: bool


class ScanSummary(WireModel):
    total_tiles: int
    tiles_with_resources: int
    tiles_with_drones: int
    tiles_with_buildings: int


class ScanData(WireModel):
    tiles: list[ScanTile]
    summary: ScanSummary


# Drone action-completed payloads.
class DriveCompletedDetails(WireModel):
    success: bool
    drive_direction: str
    difficult_terrain: bool


class IdentifyCompletedDetails(WireModel):
    success: bool
    target_q: int
    target_r: int
    distance: int
    identify_data: IdentifyData


class RepeaterActivateCompletedDetails(WireModel):
    success: bool
    active: bool
    direction: int
    transmit_range: int
    beam_width: int


class RepeaterDeactivateCompletedDetails(WireModel):
    success: bool
    active: bool


class RepeaterDeployCompletedDetails(WireModel):
    success: bool
    deployed: bool


class RepeaterRaiseCompletedDetails(WireModel):
    success: bool
    antenna_height: int
    height_change: int


class ScanCompletedDetails(WireModel):
    success: bool
    scan_range: int
    tiles_scanned: int
    scan_data: ScanData


class DroneStatusCompletedDetails(WireModel):
    success: bool
    status: DroneStatus


class TurnCompletedDetails(WireModel):
    success: bool
    turn_direction: str
    old_direction: int
    new_direction: int


ActionCompletedDetails = Union[
    DriveCompletedDetails,
    IdentifyCompletedDetails,
    RepeaterActivateCompletedDetails,
    RepeaterDeactivateCompletedDetails,
    RepeaterDeployCompletedDetails,
    RepeaterRaiseCompletedDetails,
    ScanCompletedDetails,
    DroneStatusCompletedDetails,
    TurnCompletedDetails,
]


# Drone action-queued payloads.
class CyclesRequiredDetails(WireModel):
    cycles_required: int


class LevelQueuedDetails(WireModel):
    level: int
    cycles_required: int


class MovementQueuedDetails(LevelQueuedDetails):
    direction: str


class IdentifyQueuedDetails(LevelQueuedDetails):
    target_q: int
    target_r: int


class RepeaterRaiseQueuedDetails(CyclesRequiredDetails):
    target_height: int
    height_change: int


class RepeaterTransmitQueuedDetails(CyclesRequiredDetails):
    direction: int


ActionQueuedDetails = Union[
    MovementQueuedDetails,
    IdentifyQueuedDetails,
    RepeaterRaiseQueuedDetails,
    RepeaterTransmitQueuedDetails,
    LevelQueuedDetails,
    CyclesRequiredDetails,
]


# Rejected and failed drone actions.
class ErrorDetails(WireModel):
    error: str


class MineRejectedDetails(ErrorDetails):
    resource_location: HexCoordinate


class BatteryRejectedDetails(ErrorDetails):
    required: int


ActionRejectedDetails = Union[MineRejectedDetails, BatteryRejectedDetails, ErrorDetails]


class DriveFailedDetails(ErrorDetails):
    target_q: int
    target_r: int
    building_id: str
    drone_id: str
    action_id: str


# Building payloads.
class BuildingQueuedDetails(WireModel):
    cycles: int
    efficiency: float


class BuildingStatusCompletedDetails(WireModel):
    success: bool
    status: BuildingStatus


class StatusReportSummary(WireModel):
    total_drones: int
    active_drones: int
    total_buildings: int
    active_buildings: int
    powered_buildings: int
    command_center_resources: ResourceInventory
    cycle: int


class StatusReportDrone(WireModel):
    drone_id: str
    location: HexCoordinate


class StatusReportBuilding(WireModel):
    building_id: str
    building_type: str
    origin: HexCoordinate
    coordinates: list[HexCoordinate]


class BuildingStatusReportCompletedDetails(WireModel):
    success: bool
    summary: StatusReportSummary
    drones: list[StatusReportDrone]
    buildings: list[StatusReportBuilding]
    drone_count: int
    building_count: int


BuildingActionCompletedDetails = Union[
    BuildingStatusCompletedDetails,
    BuildingStatusReportCompletedDetails,
]


# Informational payloads.
class MatchStateDetails(WireModel):
    match_state: str


class RepeaterAutoDeactivatedDetails(WireModel):
    drone_id: str
    reason: str
    battery_remaining: int


class WireMessage(WireModel):
    message_id: str
    timestamp: int
    subject: str


class ActionCompletedMessage(WireMessage):
    message_type: Literal["action_completed"]
    details: ActionCompletedDetails
    action_id: str
    drone_id: str


class ActionQueuedMessage(WireMessage):
    message_type: Literal["action_queued"]
    details: ActionQueuedDetails
    action_id: str
    drone_id: str


class ActionRejectedMessage(WireMessage):
    message_type: Literal["action_rejected"]
    details: ActionRejectedDetails
    drone_id: str


class ActionFailedMessage(WireMessage):
    message_type: Literal["action_failed"]
    details: DriveFailedDetails
    action_id: str
    drone_id: str


class BuildingActionQueuedMessage(WireMessage):
    message_type: Literal["building_action_queued"]
    details: BuildingQueuedDetails
    action_id: str
    building_id: str


class BuildingActionCompletedMessage(WireMessage):
    message_type: Literal["building_action_completed"]
    details: BuildingActionCompletedDetails
    action_id: str
    building_id: str


class InfoMessage(WireMessage):
    message_type: Literal["info"]
    details: MatchStateDetails


class WarningMessage(WireMessage):
    message_type: Literal["warning"]
    details: RepeaterAutoDeactivatedDetails
    drone_id: str


KnownWireMessage = Union[
    ActionCompletedMessage,
    ActionQueuedMessage,
    ActionRejectedMessage,
    ActionFailedMessage,
    BuildingActionQueuedMessage,
    BuildingActionCompletedMessage,
    InfoMessage,
    WarningMessage,
]


class UnknownWireMessage(WireModel):
    """An unrecognised or malformed message kept intact for later analysis."""

    message_type: str | None = None
    _raw_payload: dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def raw_payload(self) -> dict[str, Any]:
        return self._raw_payload


ParsedWireMessage = KnownWireMessage | UnknownWireMessage
