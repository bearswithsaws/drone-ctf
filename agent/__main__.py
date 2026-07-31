"""Agent entrypoint and E1.7 hello-world runtime.

Run with ``python -m agent``.  The current runtime authenticates, connects the
Socket.IO message stream, starts ActionTracker reconciliation, and repeatedly
runs the hardcoded scan/three-tiles/return proof loop on one owned drone.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

from agent.bus import EventBus
from agent.config import Config, load_config
from agent.logging_setup import setup_logging
from agent.planning.controllers.hello_world import HelloWorldController
from agent.transport.action_tracker import ActionTracker
from agent.transport.rest import GameRest
from agent.transport.ws import GameSocket

log = logging.getLogger("agent.main")


async def login(rest: GameRest, cfg: Config) -> str | None:
    """POST /auth/login and return a token, or None on failure."""
    res = await rest.post("/auth/login", json={"username": cfg.username, "password": cfg.password})
    if not res.ok:
        log.error("Login failed (%s): %s", res.status, res.error_message or res.data)
        return None
    token = res.data.get("token")
    if not token:
        log.error("Login succeeded but no token in response: %r", res.data)
        return None
    log.info("Authenticated as %s (user_id=%s)", cfg.username, res.data.get("user_id"))
    return token


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


async def run(cfg: Config, *, duration_s: float | None = None) -> int:
    log.info("Starting agent against %s", cfg.server_url)

    # A single pooled REST client used for everything. The reauth callback lets
    # GameRest transparently re-login when the server expires our token.
    rest = GameRest(cfg.api_base)

    async def reauth() -> str | None:
        return await login(rest, cfg)

    rest.set_reauth(reauth)

    socket: GameSocket | None = None
    tracker: ActionTracker | None = None
    background: list[asyncio.Task[None]] = []

    try:
        token = await login(rest, cfg)
        if token is None:
            return 1
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

        bus = EventBus()
        tracker = ActionTracker(rest, bus)
        socket = GameSocket(
            cfg.server_url,
            token_provider=lambda: rest.token,
            bus=bus,
            gap_fill=lambda count: _gap_fill(rest, count),
        )
        background = [
            asyncio.create_task(socket.run(), name="game-socket"),
            asyncio.create_task(tracker.run(), name="action-tracker"),
        ]

        await _wait_for_socket(socket)
        controller = None
        for drone_id in drone_ids:
            candidate = HelloWorldController(tracker, drone_id)
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
        if tracker is not None:
            await tracker.stop()
        if socket is not None:
            await socket.stop()
        for task in background:
            task.cancel()
        if background:
            await asyncio.gather(*background, return_exceptions=True)
        if tracker is not None:
            tracker.close()
        await rest.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent", description="Drone Battles autonomous client")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
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
        exit_code = asyncio.run(run(cfg, duration_s=args.duration))
    except KeyboardInterrupt:
        log.info("Interrupted; transport shut down")
        exit_code = 130
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
