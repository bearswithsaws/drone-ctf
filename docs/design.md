# Drone Battles Autonomous Client — Architecture

This is the durable design reference. Work is tracked as GitHub issues grouped
into milestones E1–E9. See the root `README.md` for the strategy summary.

## Why we can win

Every team starts from the same seed client (`drone-battle-client`): a Flask
*manual remote control* with full API coverage but **zero automation** — no
pathfinding, targeting, economy loops, or multi-drone coordination. We invert
that: keep the seed's hard-won knowledge (hex math, cost tables, message
semantics, wire samples) and throw away its plumbing (Flask proxies, SVG
rendering, the 4,300-line map UI), then add the intelligence it lacks.

Our edges:

- **Pipelined queues (superhuman APM).** The engine ticks ~0.25s/cycle with
  per-entity parallel FIFO queues. We keep every entity's server-side queue 4–8
  actions deep; humans clicking the seed UI act every few seconds.
- **Loss-immune command loop.** Commands and reports are silently dropped with
  probability rising to ~30% far from base. Our ActionTracker reconciles three
  signals (`action_queued`, server queue state, `action_completed`) to make
  silence unambiguous and resubmit idempotently.
- **Persistent world memory.** SQLite snapshots survive restarts and prior
  scrims; opponents reset to fog on every crash.
- **Pre-match empirical calibration.** Admin creds let us run private matches to
  measure the scoring formula, comms-loss curve, charge rates, drone cap, and
  EM/seismic signatures before match day.

## Stack decisions

| Concern | Choice | Rationale |
|---|---|---|
| Concurrency | asyncio, single event loop | Pure I/O orchestration at cycle granularity; removes the seed's thread-locking around the world model. CPU spikes (A*) go to `run_in_executor`. |
| HTTP | `httpx.AsyncClient` | Pooling + per-request timeouts; port the seed's reauth-on-401/403 logic, not its Flask coupling. |
| WebSocket | `python-socketio` AsyncClient | First-class async; port the seed's `authenticate {token}` handshake. |
| Wire models | pydantic v2 | 52 recorded samples become golden fixtures. |
| Persistence | SQLite (WAL), write-behind | Hot path stays in-memory dataclasses; one snapshot task is the only writer. |
| Commander API | FastAPI + Socket.IO ASGI (uvicorn) | Mirrors the game server's own shape, which control-plane already consumes. |

## Module layout (`agent/`)

- `config.py` — creds.txt/env loader and tunables.
- `logging_setup.py` — shared logging config.
- `__main__.py` — supervisor: wires and crash-restarts long-lived tasks.
- `bus.py` — async pub/sub event bus (replaces the seed's 130-line elif dispatch).
- `transport/` — `rest.py` (httpx wrapper, reauth), `ws.py` (socket consumer,
  reconnect, gap-fill via `GET /messages`), `action_tracker.py` (idempotency +
  loss inference — the load-bearing component), `schema.py` (pydantic models).
- `rules/` — pure functions, zero I/O: `hexmath.py` (verbatim port of the seed's
  `hex_utils.py`), `geometry.py` (elevation-aware LOS; 1 tile ≡ 5 altitude
  units), `costs.py` (ported cost/weight/consumable tables), `combat.py`,
  `economy.py`, `comms.py` (loss model, calibrated in E4).
- `sim/` — `entity_sim.py` (tracks battery/ammo the server strips from
  `action_completed`), `eta.py` (per-queue position/heading/battery prediction).
- `world/` — `model.py` + `tiles.py` (beliefs with last-seen + confidence decay),
  `tracks.py` (enemy fusion from scan/identify/seismic/DF), `threat.py` (danger
  map from weapon envelopes), `ingest.py` (message→handler registry),
  `persistence.py` (SQLite snapshot/restore).
- `planning/` — `strategist.py` (doctrine FSM), `tasks.py`, `allocator.py`
  (fitness auction + hysteresis + override pinning), `controllers/` (miner,
  scout, fighter, support, base), `pipeliner.py` (queue-saturation organ),
  `pathfind.py` (A* over terrain + threat + comms-risk + unknown costs).
- `commander/` — `api.py` (FastAPI + Socket.IO facade), `directives.py`.
- `telemetry/` — `recorder.py` (jsonl replay fuel), `metrics.py`.

Plus `packages/contract/` (versioned TS DTO/event types for control-plane, with
a pydantic→JSON-schema drift check) and `tools/` (`calibrate/`, `replay/`).

## ActionTracker (three-signal reconciliation)

Every `action_queued`/`action_completed`/`action_failed` carries a server
`action_id`. `GET /queue/drones|buildings` exposes real queue state and
`DELETE /queue/{kind}/<action_id>` / `POST /queue/reset` control it. The API
documentation says the successful POST's `action_id` identifies that queued
action, but live testing on 2026-07-31 found a server defect: the POST ID differs
from the ID shared by the queued and terminal messages. The tracker therefore
retains both as aliases and correlates a previously unknown lifecycle ID by
entity, action subject, payload where available, and FIFO order.

1. Submit → record `Intent{local_id, entity, action, preconditions, SUBMITTED}`.
2. `action_queued` → correlate via action_id, `QUEUED`.
3. No ack within `T_ack(distance)` → poll the queue and diff. An unacknowledged
   POST that is absent is treated as command loss and resubmitted *only if the
   controller-declared precondition still holds*. A successful POST is proof of
   acceptance: queue absence can mean an ID mismatch or a fast completion, so
   it is never resubmitted and is instead recorded as an inferred completion
   that a later definitive failure may correct.
4. `action_completed`/`failed` → `DONE`; feed `entity_sim` + `world.ingest`.
   Completed-without-queued ⇒ infer the queued report was also lost.
5. Every loss event feeds `metrics` and recalibrates `rules/comms.py` online.

## Runtime shape

`__main__` supervises: socket pump, tracker reconcile loop, pipeliner feed loop
(event-driven + 250ms timer), strategist tick (1–2s), snapshot loop (10s),
scoreboard poll (5s), commander uvicorn server. All crash-restarted with
backoff; state survives via SQLite. On (re)start: gap-fill `GET /messages`,
read `GET /queue/*`, full status sweep, restore snapshot, resume.

## Testing

- **Rules**: pure unit tests; golden values from `data/message_samples/*_raw.json`
  and the old-doc cost tables.
- **Ingest/replay**: `tools/replay` pipes recorded jsonl through `world.ingest`
  and asserts beliefs. Real sessions produce new fixtures for free.
- **Admin-oracle differential test**: run under fog on a private match, diff the
  world model + entity sim against admin ground truth — modeling error becomes a
  measured number.
- **Scrimmage**: agent-vs-agent (or agent-vs-seed) on the private server.
- CI runs the network-free suite; integration tests are marked `@integration`.

## Reference-client assets to port (paths under `/home/yeti/drone-battle-client`)

- `app/hex_utils.py` → `rules/hexmath.py` (verbatim).
- `app/map_store.py` constants → `rules/costs.py` (`EQUIPMENT_WEIGHTS`,
  `CONSUMABLE_SPECS`, `BUILDING_RESOURCE_CAPACITY`, `CHARGE_RATE`).
- `app/message_parser.py` `_BATTERY_COSTS`/`_estimate_battery_cost()` (~L1171),
  building cost tables, Socket.IO handshake (L136-166), per-message handler
  semantics (L607-1170) → `rules/costs.py`, `transport/ws.py`, `world/ingest.py`.
- `data/message_samples/*.json` → golden wire fixtures.
- `docs/*.md` + `*_old.md` (old docs retain dropped cost tables) → spec.
