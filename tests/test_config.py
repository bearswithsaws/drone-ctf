"""Config loader tests — pure, network-free."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import Config, _parse_creds_file, load_config

SAMPLE_CREDS = """https://example.dronebattles.test/

---

User: adminuser
Password: adminpass123

Admin ID: abc-123

API Token: deadbeefcafe

Team 1\t40A0285FF9BDB9AC1742E3BF

---

Team 1, User1:

username: user1
password: userpass
User ID: d85761ed-3078-42f8-9b32-b3f7031a3a40
"""


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "creds.txt"
    p.write_text(SAMPLE_CREDS, encoding="utf-8")
    return p


def test_parse_creds_file(tmp_path: Path) -> None:
    creds = _parse_creds_file(_write(tmp_path))
    assert creds["server_url"] == "https://example.dronebattles.test/"
    assert creds["admin_username"] == "adminuser"
    assert creds["admin_password"] == "adminpass123"
    assert creds["admin_token"] == "deadbeefcafe"
    assert creds["username"] == "user1"
    assert creds["password"] == "userpass"


def test_load_config_from_creds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "DRONE_SERVER_URL",
        "DRONE_USERNAME",
        "DRONE_PASSWORD",
        "DRONE_ADMIN_USERNAME",
        "DRONE_ADMIN_PASSWORD",
        "DRONE_ADMIN_TOKEN",
        "DRONE_TELEMETRY_PATH",
        "DRONE_WORLD_DATABASE",
        "DRONE_MATCH_ID",
        "DRONE_PLANNING_ENABLED",
        "DRONE_QUEUE_DEPTH_TARGET",
        "DRONE_STRATEGIST_TICK_S",
        "DRONE_SNAPSHOT_INTERVAL_S",
        "DRONE_SCOREBOARD_POLL_S",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = load_config(_write(tmp_path))
    assert isinstance(cfg, Config)
    assert cfg.api_base == "https://example.dronebattles.test/api/v1"
    assert cfg.username == "user1"
    assert cfg.admin_token == "deadbeefcafe"
    assert cfg.persistence_match_id.startswith("runtime-")


def test_env_overrides_creds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRONE_USERNAME", "envuser")
    monkeypatch.setenv("DRONE_PASSWORD", "envpass")
    cfg = load_config(_write(tmp_path))
    assert cfg.username == "envuser"
    assert cfg.password == "envpass"
    # server_url still comes from the creds file
    assert cfg.server_url == "https://example.dronebattles.test/"


def test_missing_config_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("DRONE_SERVER_URL", "DRONE_USERNAME", "DRONE_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    empty = tmp_path / "creds.txt"
    empty.write_text("nothing useful here\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Missing required config"):
        load_config(empty)


def test_runtime_settings_load_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DRONE_TELEMETRY_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("DRONE_WORLD_DATABASE", str(tmp_path / "world.sqlite"))
    monkeypatch.setenv("DRONE_MATCH_ID", "match-77")
    monkeypatch.setenv("DRONE_PLANNING_ENABLED", "true")
    monkeypatch.setenv("DRONE_QUEUE_DEPTH_TARGET", "8")
    monkeypatch.setenv("DRONE_STRATEGIST_TICK_S", "0.75")
    monkeypatch.setenv("DRONE_SNAPSHOT_INTERVAL_S", "2.5")
    monkeypatch.setenv("DRONE_SCOREBOARD_POLL_S", "3")

    cfg = load_config(_write(tmp_path))

    assert cfg.telemetry_path == tmp_path / "events.jsonl"
    assert cfg.world_database == tmp_path / "world.sqlite"
    assert cfg.persistence_match_id == "match-77"
    assert cfg.planning_enabled
    assert cfg.queue_depth_target == 8
    assert cfg.strategist_tick_s == 0.75
    assert cfg.snapshot_interval_s == 2.5
    assert cfg.scoreboard_poll_s == 3


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("DRONE_PLANNING_ENABLED", "sometimes", "must be a boolean"),
        ("DRONE_QUEUE_DEPTH_TARGET", "9", "must be between 4 and 8"),
        ("DRONE_STRATEGIST_TICK_S", "0", "must be at least"),
        ("DRONE_SNAPSHOT_INTERVAL_S", "0", "must be at least"),
        ("DRONE_SCOREBOARD_POLL_S", "0", "must be at least"),
    ],
)
def test_invalid_runtime_settings_fail_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=message):
        load_config(_write(tmp_path))
