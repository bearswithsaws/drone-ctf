"""Agent entrypoint and task supervisor.

Run with ``python -m agent``.

At this stage (E1.1) the supervisor does the minimum walking-skeleton work:
load config, authenticate against the game server, report success, exit cleanly.
Long-lived supervised tasks (socket pump, action-tracker reconcile loop,
pipeliner, strategist tick, persistence snapshots, scoreboard poll, commander
API) are wired in as they land in later tickets — see docs/design.md.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import httpx

from agent.config import Config, load_config
from agent.logging_setup import setup_logging

log = logging.getLogger("agent.main")


async def authenticate(cfg: Config) -> str:
    """Log in and return an auth token. Raises on failure.

    Superseded by the full REST transport in E1.2 (transport/rest.py); kept here
    so the skeleton has an end-to-end auth path from day one.
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
        resp = await client.post(
            f"{cfg.api_base}/auth/login",
            json={"username": cfg.username, "password": cfg.password},
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("token")
        if not token:
            raise RuntimeError(f"Login succeeded but no token in response: {data!r}")
        log.info("Authenticated as %s (user_id=%s)", cfg.username, data.get("user_id"))
        return token


async def run(cfg: Config) -> int:
    log.info("Starting agent against %s", cfg.server_url)
    try:
        await authenticate(cfg)
    except httpx.HTTPError as exc:
        log.error("Authentication failed: %s", exc)
        return 1

    # TODO(E1.x+): construct EventBus, GameRest, GameSocket, ActionTracker,
    # WorldModel, Strategist, Pipeliner, commander API; supervise them with
    # crash-restart + backoff here. For now the skeleton exits after auth.
    log.info("Walking-skeleton startup complete; exiting cleanly.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent", description="Drone Battles autonomous client")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config()
    raise SystemExit(asyncio.run(run(cfg)))


if __name__ == "__main__":
    main()
