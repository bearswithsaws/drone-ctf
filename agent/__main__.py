"""Agent entrypoint: the playable composed runtime, plus the E1.7 proof mode.

Run with ``python -m agent`` (default ``--mode play``): the composition root
owns transport, telemetry, world state, simulation, and persistence, and the
strategist is installed into the planning boundary so the agent autonomously
mines, researches, produces, and defends. ``--mode proof`` runs the finite
hello-world out-and-back controller instead.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

from agent.config import Config, load_config
from agent.logging_setup import setup_logging
from agent.planning.controllers.hello_world import HelloWorldController
from agent.planning.live import build_strategist
from agent.planning.strategist import StrategistConfig
from agent.runtime import AgentRuntime, compose_runtime
from agent.transport.action_tracker import ActionTracker
from agent.transport.rest import GameRest
from agent.transport.ws import GameSocket

log = logging.getLogger("agent.main")


async def login(rest: GameRest, cfg: Config) -> tuple[str, str] | None:
    """POST /auth/login; return ``(token, user_id)`` or None on failure."""
    res = await rest.post("/auth/login", json={"username": cfg.username, "password": cfg.password})
    if not res.ok:
        log.error("Login failed (%s): %s", res.status, res.error_message or res.data)
        return None
    token = res.data.get("token")
    if not token:
        log.error("Login succeeded but no token in response: %r", res.data)
        return None
    user_id = str(res.data.get("user_id") or "")
    log.info("Authenticated as %s (user_id=%s)", cfg.username, user_id)
    return token, user_id


def _extract_drone_ids(payload: dict[str, Any]) -> list[str]:
    """Accept the documented response plus older private-server variants."""

    raw = payload.get("drone_ids", payload.get("drones", []))
    if not isinstance(raw, list):
        return []

    found: list[str] = []
    for item in raw:
        drone_id: Any = item
        if isinstance(item, dict):
            drone_id = item.get("drone_id", item.get("id"))
        if isinstance(drone_id, str) and drone_id and drone_id not in found:
            found.append(drone_id)
    return found


async def _gap_fill(rest: GameRest, count: int) -> list[dict[str, Any]]:
    result = await rest.get("/messages", params={"count": count, "consume": False})
    if not result.ok:
        raise RuntimeError(
            f"GET /messages failed ({result.status}): {result.error_message or result.data}"
        )
    messages = result.data.get("messages", [])
    if not isinstance(messages, list):
        raise RuntimeError("GET /messages returned a non-list 'messages' field")
    return [message for message in messages if isinstance(message, dict)]


async def _wait_for_socket(socket: GameSocket, timeout_s: float = 30.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not socket.is_connected:
        if loop.time() >= deadline:
            raise TimeoutError(f"Socket.IO authentication did not complete within {timeout_s:g}s")
        await asyncio.sleep(0.05)


async def run(cfg: Config, *, mode: str = "play", duration_s: float | None = None) -> int:
    log.info("Starting agent against %s (mode=%s)", cfg.server_url, mode)

    # A single pooled REST client used for everything. The reauth callback lets
    # GameRest transparently re-login when the server expires our token.
    rest = GameRest(cfg.api_base)

    async def reauth() -> str | None:
        result = await login(rest, cfg)
        return None if result is None else result[0]

    rest.set_reauth(reauth)

    runtime: AgentRuntime | None = None

    try:
        auth = await login(rest, cfg)
        if auth is None:
            return 1
        token, user_id = auth
        rest.set_token(token)

        drones = await rest.get("/drones")
        if not drones.ok:
            log.error(
                "Drone discovery failed (%s): %s",
                drones.status,
                drones.error_message or drones.data,
            )
            return 1
        drone_ids = _extract_drone_ids(drones.data)
        if not drone_ids:
            log.error("No owned drones were returned by GET /drones")
            return 1

        strategy = None
        if mode == "play":
            strategy = build_strategist(
                rest,
                user_id,
                config=StrategistConfig(tick_interval_s=cfg.strategist_tick_s),
                scoreboard_interval_s=cfg.scoreboard_poll_s,
            )

        runtime = compose_runtime(
            cfg,
            rest,
            gap_fill=lambda count: _gap_fill(rest, count),
            tracker_factory=ActionTracker,
            socket_factory=GameSocket,
            strategy=strategy,
        )
        await runtime.start()
        await _wait_for_socket(runtime.socket)

        if mode == "play":
            # Prime the world so the strategist has drones/buildings/resources to
            # reason about, then let the composed subsystems play until stopped.
            report = await rest.post(
                "/buildings/command_centre/status_report", json={"efficiency": 1.0}
            )
            if not report.ok:
                log.warning(
                    "Initial status_report request failed (%s): %s — strategist "
                    "will wait for the first successful report",
                    report.status,
                    report.error_message or report.data,
                )
            log.info(
                "Playing autonomously with %d known drone(s)%s",
                len(drone_ids),
                f" for {duration_s:g}s" if duration_s else " until interrupted",
            )
            if duration_s is None:
                await asyncio.Event().wait()  # play until Ctrl-C
            else:
                await asyncio.sleep(duration_s)
            return 0

        controller = None
        for drone_id in drone_ids:
            candidate = HelloWorldController(runtime.tracker, drone_id)
            if await candidate.prepare():
                controller = candidate
                break
        if controller is None:
            log.error("No owned drone has a verified clear three-tile out-and-back route")
            return 1
        log.info(
            "Transport ready; selected drone %s with a verified clear route",
            controller.drone_id,
        )
        loops = await controller.run(duration_s=duration_s)
        log.info("Hello-world autonomy stopped cleanly after %d loop(s)", loops)
        return 0
    finally:
        if runtime is not None:
            await runtime.stop()
        else:
            await rest.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent", description="Drone Battles autonomous client")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    parser.add_argument(
        "--mode",
        choices=("play", "proof"),
        default="play",
        help="play = autonomous strategist-driven runtime (default); "
        "proof = finite E1.7 hello-world out-and-back",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="stop after this many seconds, after the current action loop reaches terminal states",
    )
    args = parser.parse_args()

    if args.duration is not None and args.duration < 0:
        parser.error("--duration cannot be negative")

    setup_logging(args.log_level)
    cfg = load_config()
    try:
        exit_code = asyncio.run(run(cfg, mode=args.mode, duration_s=args.duration))
    except KeyboardInterrupt:
        log.info("Interrupted; transport shut down")
        exit_code = 130
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
