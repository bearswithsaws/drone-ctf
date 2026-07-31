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

from agent.config import Config, load_config
from agent.logging_setup import setup_logging
from agent.transport.rest import GameRest

log = logging.getLogger("agent.main")


async def login(rest: GameRest, cfg: Config) -> str | None:
    """POST /auth/login and return a token, or None on failure."""
    res = await rest.post(
        "/auth/login", json={"username": cfg.username, "password": cfg.password}
    )
    if not res.ok:
        log.error("Login failed (%s): %s", res.status, res.error_message or res.data)
        return None
    token = res.data.get("token")
    if not token:
        log.error("Login succeeded but no token in response: %r", res.data)
        return None
    log.info("Authenticated as %s (user_id=%s)", cfg.username, res.data.get("user_id"))
    return token


async def run(cfg: Config) -> int:
    log.info("Starting agent against %s", cfg.server_url)

    # A single pooled REST client used for everything. The reauth callback lets
    # GameRest transparently re-login when the server expires our token.
    rest = GameRest(cfg.api_base)

    async def reauth() -> str | None:
        return await login(rest, cfg)

    rest.set_reauth(reauth)

    try:
        token = await login(rest, cfg)
        if token is None:
            return 1
        rest.set_token(token)

        # TODO(E1.x+): construct EventBus, GameSocket, ActionTracker, WorldModel,
        # Strategist, Pipeliner, commander API; supervise them with crash-restart
        # + backoff here. For now the skeleton exits cleanly after auth.
        log.info("Walking-skeleton startup complete; exiting cleanly.")
        return 0
    finally:
        await rest.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent", description="Drone Battles autonomous client")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config()
    raise SystemExit(asyncio.run(run(cfg)))


if __name__ == "__main__":
    main()
