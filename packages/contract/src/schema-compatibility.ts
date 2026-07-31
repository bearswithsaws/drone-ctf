/**
 * Compile-time proof that the hand-written public types are structurally
 * equivalent to the types generated from the Pydantic JSON Schema.
 */
import type * as Public from "./index.js";
import type * as Schema from "./schema.generated.js";

type Equivalent<Left, Right> =
  [Left] extends [Right] ? ([Right] extends [Left] ? true : false) : false;
type Assert<Value extends true> = Value;

export type SchemaCompatibility = [
  Assert<Equivalent<Public.Coordinate, Schema.Coordinate>>,
  Assert<Equivalent<Public.TileState, Schema.TileState>>,
  Assert<Equivalent<Public.DroneState, Schema.DroneState>>,
  Assert<Equivalent<Public.BuildingState, Schema.BuildingState>>,
  Assert<Equivalent<Public.EnemyTrackState, Schema.EnemyTrackState>>,
  Assert<Equivalent<Public.StateSnapshot, Schema.StateSnapshot>>,
  Assert<Equivalent<Public.TileUpsert, Schema.TileUpsert>>,
  Assert<Equivalent<Public.DroneUpsert, Schema.DroneUpsert>>,
  Assert<Equivalent<Public.BuildingUpsert, Schema.BuildingUpsert>>,
  Assert<Equivalent<Public.EnemyTrackUpsert, Schema.EnemyTrackUpsert>>,
  Assert<Equivalent<Public.TileRemoved, Schema.TileRemoved>>,
  Assert<Equivalent<Public.EntityRemoved, Schema.EntityRemoved>>,
  Assert<Equivalent<Public.WorldChange, Schema.WorldDiff["changes"][number]>>,
  Assert<Equivalent<Public.WorldDiff, Schema.WorldDiff>>,
  Assert<Equivalent<Public.PlanTask, Schema.PlanTask>>,
  Assert<Equivalent<Public.PlanAssignment, Schema.PlanAssignment>>,
  Assert<Equivalent<Public.PlannedPath, Schema.PlannedPath>>,
  Assert<Equivalent<Public.PlanSnapshot, Schema.PlanSnapshot>>,
  Assert<Equivalent<Public.StanceDirective, Schema.StanceDirective>>,
  Assert<Equivalent<Public.SquadOrderDirective, Schema.SquadOrderDirective>>,
  Assert<Equivalent<Public.EntityOverrideDirective, Schema.EntityOverrideDirective>>,
  Assert<
    Equivalent<
      Public.Directive,
      Schema.StanceDirective | Schema.SquadOrderDirective | Schema.EntityOverrideDirective
    >
  >,
  Assert<Equivalent<Public.Alert, Schema.Alert>>,
  Assert<Equivalent<Public.WorldDiffEvent, Schema.WorldDiffEvent>>,
  Assert<Equivalent<Public.PlanChangedEvent, Schema.PlanChangedEvent>>,
  Assert<Equivalent<Public.AlertEvent, Schema.AlertEvent>>,
  Assert<
    Equivalent<
      Public.SocketEvent,
      Schema.WorldDiffEvent | Schema.PlanChangedEvent | Schema.AlertEvent
    >
  >,
  Assert<Equivalent<Public.ContractDocument, Schema.DroneCTFCommanderContractV1>>,
];
