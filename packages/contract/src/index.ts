/** Version carried by every snapshot and diff in package major version 1. */
export const CONTRACT_VERSION = "1.0" as const;

export type ContractVersion = typeof CONTRACT_VERSION;

export interface Coordinate {
  q: number;
  r: number;
}

export type TileSource = "scan" | "identify" | "status_report" | "admin";

export interface TileState {
  coordinate: Coordinate;
  terrain_type: number | null;
  elevation: number | null;
  has_resource: boolean;
  resource_type: string | null;
  resource_amount: number | null;
  has_building: boolean;
  building_id: string | null;
  has_drone: boolean;
  last_seen: number;
  source: TileSource;
  confidence: number;
}

export interface DroneState {
  drone_id: string;
  coordinate: Coordinate | null;
  direction: number | null;
  elevation: number;
  last_seen: number;
}

export type BuildingAllegiance = "friendly" | "enemy";

export interface BuildingState {
  building_id: string;
  building_type: string | null;
  allegiance: BuildingAllegiance;
  origin: Coordinate | null;
  tiles: Coordinate[];
  last_seen: number;
}

export type EnemyTrackSource = "scan" | "identify" | "seismic" | "df_antenna";

export interface EnemyTrackState {
  track_id: string;
  drone_id: string | null;
  coordinate: Coordinate;
  predicted_coordinate: Coordinate;
  first_seen: number;
  last_seen: number;
  source: EnemyTrackSource;
  confidence: number;
  is_decoy: boolean;
  equipment: string[];
  equipment_known: boolean;
}

export interface StateSnapshot {
  contract_version: ContractVersion;
  sequence: number;
  cycle: number;
  /** RFC 3339 date-time. */
  generated_at: string;
  tiles: TileState[];
  drones: DroneState[];
  buildings: BuildingState[];
  enemy_tracks: EnemyTrackState[];
}

export interface TileUpsert {
  type: "tile_upsert";
  tile: TileState;
}

export interface DroneUpsert {
  type: "drone_upsert";
  drone: DroneState;
}

export interface BuildingUpsert {
  type: "building_upsert";
  building: BuildingState;
}

export interface EnemyTrackUpsert {
  type: "enemy_track_upsert";
  enemy_track: EnemyTrackState;
}

export interface TileRemoved {
  type: "tile_removed";
  coordinate: Coordinate;
}

export interface EntityRemoved {
  type: "entity_removed";
  entity_type: "drone" | "building" | "enemy_track";
  entity_id: string;
}

export type WorldChange =
  | TileUpsert
  | DroneUpsert
  | BuildingUpsert
  | EnemyTrackUpsert
  | TileRemoved
  | EntityRemoved;

export interface WorldDiff {
  contract_version: ContractVersion;
  /** Monotonic sequence; the first diff after a snapshot is snapshot.sequence + 1. */
  sequence: number;
  cycle: number;
  changes: WorldChange[];
}

export type Scalar = string | number | boolean | null;

export interface PlanTask {
  task_id: string;
  kind: string;
  priority: number;
  max_assignees: number;
  parameters: Record<string, Scalar>;
}

export interface PlanAssignment {
  entity_id: string;
  task_id: string;
  fitness: number | null;
  pinned: boolean;
  retained: boolean;
}

export interface PlannedPath {
  entity_id: string;
  coordinates: Coordinate[];
}

export interface PlanSnapshot {
  contract_version: ContractVersion;
  sequence: number;
  cycle: number;
  tasks: PlanTask[];
  assignments: PlanAssignment[];
  paths: PlannedPath[];
}

export interface DirectiveFields {
  directive_id: string;
  /** RFC 3339 date-time. */
  issued_at: string;
  ttl_seconds: number;
}

export interface StanceDirective extends DirectiveFields {
  kind: "stance";
  stance: "aggressive" | "balanced" | "defensive" | "retreat";
}

export interface SquadOrderDirective extends DirectiveFields {
  kind: "squad_order";
  squad_id: string;
  order:
    | "move"
    | "hold"
    | "mine"
    | "scout"
    | "escort"
    | "strike"
    | "lay_mines"
    | "deploy_repeater";
  target: Coordinate | string | null;
}

export interface EntityOverrideDirective extends DirectiveFields {
  kind: "entity_override";
  entity_id: string;
  task_id: string;
}

export type Directive =
  | StanceDirective
  | SquadOrderDirective
  | EntityOverrideDirective;

export interface Alert {
  alert_id: string;
  severity: "info" | "warning" | "critical";
  code: string;
  message: string;
  cycle: number;
  /** RFC 3339 date-time. */
  created_at: string;
  entity_ids: string[];
  coordinate: Coordinate | null;
}

export interface WorldDiffEvent {
  event: "world_diff";
  data: WorldDiff;
}

export interface PlanChangedEvent {
  event: "plan_changed";
  data: PlanSnapshot;
}

export interface AlertEvent {
  event: "alert";
  data: Alert;
}

export type SocketEvent = WorldDiffEvent | PlanChangedEvent | AlertEvent;

export type ContractDocument =
  | StateSnapshot
  | WorldDiff
  | PlanSnapshot
  | Directive
  | SocketEvent;
