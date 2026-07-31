import {
  CONTRACT_VERSION,
  type Directive,
  type SocketEvent,
  type StateSnapshot,
} from "../src/index.js";

const snapshot: StateSnapshot = {
  contract_version: CONTRACT_VERSION,
  sequence: 0,
  cycle: 0,
  generated_at: "2026-07-31T12:00:00Z",
  tiles: [],
  drones: [],
  buildings: [],
  enemy_tracks: [],
};

const directive: Directive = {
  kind: "entity_override",
  directive_id: "directive-1",
  issued_at: "2026-07-31T12:00:00Z",
  ttl_seconds: 30,
  entity_id: "drone-1",
  task_id: "scout-north",
};

const event: SocketEvent = {
  event: "world_diff",
  data: {
    contract_version: CONTRACT_VERSION,
    sequence: snapshot.sequence + 1,
    cycle: snapshot.cycle,
    changes: [],
  },
};

void directive;
void event;
