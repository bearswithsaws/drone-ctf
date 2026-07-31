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

    @property
    def api_base(self) -> str:
        return self.server_url.rstrip("/") + "/api/v1"


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
      DRONE_ADMIN_USERNAME, DRONE_ADMIN_PASSWORD, DRONE_ADMIN_TOKEN
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
    )
