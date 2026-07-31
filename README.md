# drone-ctf

Autonomous competitive client for the **Drone Battles** hex-grid RTS.

Every team in the competition starts from the same seed client
(`drone-battle-client`) — a manual remote control with full API coverage but no
automation. This project builds an autonomous agent that outperforms those
derivatives through superhuman action-queue throughput, a loss-immune command
loop, persistent world memory, threat-aware pathing, and pre-match empirical
calibration on a private server.

- **Architecture:** [`docs/design.md`](docs/design.md)
- **Work tracking:** GitHub Issues grouped into milestones **E1–E9** (epics).
- **Strategy in one line:** keep the seed's knowledge (hex math, cost tables,
  message semantics), throw away its plumbing, add the intelligence it lacks —
  economy autopilot, planning/allocator, combat micro, and a commander API that
  the `control-plane` 3D UI drives.

## Layout

```
agent/           Python agent core (asyncio)
  transport/     REST + Socket.IO + ActionTracker
  rules/         pure game-rule functions (ported from the seed client)
  sim/           local simulation of server-stripped state
  world/         world model, enemy tracks, threat map, persistence
  planning/      strategist, allocator, controllers, pipeliner, pathfinding
  commander/     FastAPI + Socket.IO facade for the UI
  telemetry/     recorder + metrics
packages/contract/  versioned TS types shared with control-plane
tools/           calibrate/ (private-match scripts), replay/ (jsonl harness)
tests/           network-free unit tests (CI)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                          # network-free unit tests
python -m agent                 # run the autonomous match loop until match end/interruption
python -m agent --duration 600  # finite 10-minute autonomous run
python -m agent --mode proof    # explicit E1.7 hello-world transport proof
```

Recorded telemetry can be replayed through the world model and entity simulator
and checked against a known final state:

```bash
tools/replay tests/fixtures/replay_session.jsonl \
  --expected tests/fixtures/replay_expected.json
```

Credentials and the server URL come from environment variables
(`DRONE_SERVER_URL`, `DRONE_USERNAME`, `DRONE_PASSWORD`, and optional
`DRONE_ADMIN_*`) or from a gitignored `creds.txt` at the repo root.
**`creds.txt` is never committed.**

The live composition records raw messages to `telemetry/live.jsonl`, restores
and snapshots the world model in `state/world.sqlite`, and feeds each parsed
message to action tracking, world ingest/enemy tracks, entity simulation, and
metrics. `DRONE_TELEMETRY_PATH`, `DRONE_WORLD_DATABASE`, and `DRONE_MATCH_ID`
override those persistence settings. Production mode clears stale owned queues,
runs a player-visible command-centre/status/scan sweep, polls the scoreboard,
and continuously turns strategist tasks into miner, scout, fighter, refining,
research, and production controller plans.

The commander facade exposes `GET /v1/state` plus `POST /v1/directives` for
time-limited stance, squad-order, and per-entity override commands. Entity
overrides pin the selected allocator task across replans until their absolute
TTL expires; stance changes wake the live strategist immediately.

In `--mode proof`, the E1.7 proof controller selects the first owned drone with
a clear route, then repeatedly scans, drives forward three tiles, and reverses
home. A finite `--duration` is checked between complete loops, so shutdown never
leaves a locally submitted action in flight. Shutdown stops live producers,
flushes a final world snapshot, closes the JSONL session, and joins every
runtime task.
