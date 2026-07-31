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
python -m agent                 # run hello-world autonomy until interrupted
python -m agent --duration 600  # finite 10-minute E1.7 acceptance run
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
override those persistence settings. The pipeliner and allocator are present
behind a strategy boundary; `DRONE_PLANNING_ENABLED` starts the empty planning
service, but no unfinished strategic controllers are installed by default.

The E1.7 proof controller still selects the first owned drone with a clear
route, then repeatedly scans, drives forward three tiles, and reverses home. A
finite `--duration` is checked between complete loops, so shutdown never leaves
a locally submitted action in flight. Shutdown stops live producers, flushes a
final world snapshot, closes the JSONL session, and joins every runtime task.

For unattended operation, [`deploy/drone-agent.service`](deploy/drone-agent.service)
restarts the process after a crash or `SIGKILL`. Long-running asyncio services
are also supervised independently with bounded exponential backoff, and every
process start restores its snapshot, gap-fills messages, reads both action
queues, and requests an authoritative command-center status report before
planning begins. See [`docs/operations.md`](docs/operations.md) for installation
and the under-30-second restart acceptance check.
