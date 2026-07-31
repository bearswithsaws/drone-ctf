"""Runtime configuration.

Credentials and the server URL are read from environment variables first, then
from a ``creds.txt`` file at the repo root as a fallback. ``creds.txt`` is
gitignored and must never be committed.

The ``creds.txt`` format is a loose ``key: value`` / labelled block layout (see
the file in the repo root). We parse only what we need and tolerate the rest.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CREDS = REPO_ROOT / "creds.txt"


@dataclass(frozen=True)
class Config:
    server_url: str
    username: str
    password: str
    # Optional admin creds — only needed for private-match calibration (E4).
    admin_username: str | None = None
    admin_password: str | None = None
    admin_token: str | None = None
    # Tunables.
    queue_depth_target: int = 6
    strategist_tick_s: float = 1.5
    snapshot_interval_s: float = 10.0
    scoreboard_poll_s: float = 5.0
    telemetry_path: Path = REPO_ROOT / "telemetry" / "live.jsonl"
    world_database: Path = REPO_ROOT / "state" / "world.sqlite"
    match_id: str | None = None
    planning_enabled: bool = False

    @property
    def api_base(self) -> str:
        return self.server_url.rstrip("/") + "/api/v1"

    @property
    def persistence_match_id(self) -> str:
        """Stable snapshot key, overridable when the server exposes a match id."""

        if self.match_id:
            return self.match_id
        identity = f"{self.server_url.rstrip('/')}\0{self.username}".encode()
        return f"runtime-{sha256(identity).hexdigest()[:16]}"


def _parse_creds_file(path: Path) -> dict[str, str]:
    """Extract the values we care about from the labelled creds.txt.

    Returns a dict with any of: server_url, username, password,
    admin_username, admin_password, admin_token.
    """
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}

    # First non-comment URL line is the server.
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("http://") or s.startswith("https://"):
            out["server_url"] = s
            break

    # Admin block: "User: adam" / "Password: ..." appear before the team section.
    m = re.search(r"^User:\s*(\S+)", text, re.MULTILINE)
    if m:
        out["admin_username"] = m.group(1)
    # The admin Password line is the first "Password:" in the file.
    m = re.search(r"^Password:\s*(\S+)", text, re.MULTILINE)
    if m:
        out["admin_password"] = m.group(1)
    m = re.search(r"^API Token:\s*(\S+)", text, re.MULTILINE)
    if m:
        out["admin_token"] = m.group(1)

    # Team user block: "username: user1" / "password: password" (lowercase keys).
    m = re.search(r"^username:\s*(\S+)", text, re.MULTILINE)
    if m:
        out["username"] = m.group(1)
    m = re.search(r"^password:\s*(\S+)", text, re.MULTILINE)
    if m:
        out["password"] = m.group(1)

    return out


def load_config(creds_path: Path | None = None) -> Config:
    """Build a Config from env vars, falling back to creds.txt.

    Env vars (all optional if creds.txt supplies them):
      DRONE_SERVER_URL, DRONE_USERNAME, DRONE_PASSWORD,
      DRONE_ADMIN_USERNAME, DRONE_ADMIN_PASSWORD, DRONE_ADMIN_TOKEN,
      DRONE_TELEMETRY_PATH, DRONE_WORLD_DATABASE, DRONE_MATCH_ID,
      DRONE_PLANNING_ENABLED, DRONE_QUEUE_DEPTH_TARGET,
      DRONE_SNAPSHOT_INTERVAL_S
    """
    creds = _parse_creds_file(creds_path or DEFAULT_CREDS)

    def pick(env_key: str, creds_key: str) -> str | None:
        return os.environ.get(env_key) or creds.get(creds_key)

    server_url = pick("DRONE_SERVER_URL", "server_url")
    username = pick("DRONE_USERNAME", "username")
    password = pick("DRONE_PASSWORD", "password")

    missing = [
        name
        for name, val in (
            ("server_url", server_url),
            ("username", username),
            ("password", password),
        )
        if not val
    ]
    if missing:
        raise RuntimeError(
            "Missing required config: "
            + ", ".join(missing)
            + ". Provide via env (DRONE_SERVER_URL/DRONE_USERNAME/DRONE_PASSWORD) "
            "or creds.txt at the repo root."
        )

    return Config(
        server_url=server_url,  # type: ignore[arg-type]
        username=username,  # type: ignore[arg-type]
        password=password,  # type: ignore[arg-type]
        admin_username=pick("DRONE_ADMIN_USERNAME", "admin_username"),
        admin_password=pick("DRONE_ADMIN_PASSWORD", "admin_password"),
        admin_token=pick("DRONE_ADMIN_TOKEN", "admin_token"),
        queue_depth_target=_env_int("DRONE_QUEUE_DEPTH_TARGET", 6, minimum=4, maximum=8),
        snapshot_interval_s=_env_float("DRONE_SNAPSHOT_INTERVAL_S", 10.0, minimum=0.001),
        telemetry_path=Path(
            os.environ.get("DRONE_TELEMETRY_PATH", REPO_ROOT / "telemetry" / "live.jsonl")
        ),
        world_database=Path(
            os.environ.get("DRONE_WORLD_DATABASE", REPO_ROOT / "state" / "world.sqlite")
        ),
        match_id=os.environ.get("DRONE_MATCH_ID") or None,
        planning_enabled=_env_bool("DRONE_PLANNING_ENABLED", False),
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean (true/false)")


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return value


def _env_float(name: str, default: float, *, minimum: float) -> float:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum:g}")
    return value
