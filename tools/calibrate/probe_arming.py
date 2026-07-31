"""HUNT-T1: empirically probe the real arming path with a player token.

Answers the plan's open questions by calling the live endpoints on our own
team's assets and capturing the server's messages:

1. Does `POST /drones/<id>/equipment {equipment_type}` work for a player, or
   error / require admin?
2. Does `production/manufacture` at the equipment plant consume the plant's own
   stored resources (=> we must haul refined resources there) or the CC's? What
   does it do with zero resources?
3. `equipment_plant/install` coordinate frame — absolute, or relative to the
   plant origin? Adjacency requirement? Error shape.
4. Do the production buildings hold any resources at all right now?

Run:  python -m tools.calibrate.probe_arming
It logs in from creds.txt/env, snapshots buildings+drones via status_report,
fires each probe, and prints every correlated server message. Read-only where
possible; the POST probes act only on our own team and are individually logged.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from agent.config import load_config
from agent.logging_setup import setup_logging
from agent.bus import EventBus
from agent.transport.rest import GameRest
from agent.transport.ws import TOPIC_MESSAGE, GameSocket

log = logging.getLogger("probe.arming")


class Collector:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def on_message(self, _topic: str, msg: Any) -> None:
        if isinstance(msg, dict):
            self.messages.append(msg)

    def drain(self) -> list[dict[str, Any]]:
        out = self.messages[:]
        self.messages.clear()
        return out


def _pick(status_report: dict[str, Any]) -> dict[str, Any]:
    """Extract equipment plant / drone plant / a drone from a status_report details."""
    buildings = status_report.get("buildings", []) or []
    drones = status_report.get("drones", []) or []
    by_type = {}
    for b in buildings:
        by_type.setdefault(b.get("building_type"), b)
    return {
        "equipment_plant": by_type.get("drone_equipment_production_plant"),
        "drone_plant": by_type.get("drone_production_plant"),
        "command_center": by_type.get("command_center"),
        "buildings": buildings,
        "drone": drones[0] if drones else None,
        "drones": drones,
    }


async def _await_messages(collector: Collector, seconds: float) -> list[dict[str, Any]]:
    await asyncio.sleep(seconds)
    return collector.drain()


def _dump(label: str, messages: list[dict[str, Any]]) -> None:
    print(f"\n=== {label}: {len(messages)} message(s) ===")
    for m in messages:
        print(
            json.dumps(
                {
                    "type": m.get("message_type"),
                    "subject": m.get("subject"),
                    "details": m.get("details"),
                    "action_id": m.get("action_id"),
                    "building_id": m.get("building_id"),
                    "drone_id": m.get("drone_id"),
                },
                default=str,
            )
        )


async def main() -> None:
    setup_logging("INFO")
    cfg = load_config()
    rest = GameRest(cfg.api_base)
    res = await rest.post("/auth/login", json={"username": cfg.username, "password": cfg.password})
    assert res.ok, res.data
    rest.set_token(res.data["token"])

    collector = Collector()
    bus = EventBus()
    bus.subscribe(TOPIC_MESSAGE, collector.on_message)

    async def gap_fill(n: int) -> list[dict[str, Any]]:
        r = await rest.get("/messages", params={"count": n, "consume": False})
        return r.data.get("messages", []) if r.ok else []

    socket = GameSocket(cfg.server_url, token_provider=lambda: rest.token, bus=bus, gap_fill=gap_fill)
    sock_task = asyncio.create_task(socket.run())

    try:
        for _ in range(60):
            if socket.is_connected:
                break
            await asyncio.sleep(0.25)
        collector.drain()

        # --- snapshot: buildings, drones, stored resources ---
        rep = await rest.post("/buildings/command_centre/status_report", json={"efficiency": 1.0})
        print(f"\n[status_report POST] ok={rep.ok} status={rep.status} data={rep.data}")
        msgs = await _await_messages(collector, 3.0)
        report = next(
            (m for m in msgs if m.get("message_type") == "building_action_completed"
             and "status_report" in (m.get("subject") or "").lower()),
            None,
        )
        if report is None:
            print("!! no status_report message arrived; aborting probe")
            return
        picked = _pick(report.get("details") or {})

        for key in ("equipment_plant", "drone_plant", "command_center"):
            b = picked[key]
            if b is None:
                print(f"\n[{key}] NOT FOUND")
                continue
            print(
                f"\n[{key}] id={b.get('building_id')} origin={b.get('origin')} "
                f"stored={b.get('stored_resources')}"
            )

        drone = picked["drone"]
        equip_plant = picked["equipment_plant"]
        if drone is None:
            print("!! no drones in report; aborting")
            return
        drone_id = drone.get("drone_id")
        drone_loc = drone.get("location")
        print(f"\n[probe drone] id={drone_id} location(abs)={drone_loc}")

        # Full drone status (equipment + free slots)
        dstat = await rest.post(f"/drones/{drone_id}/status", json={})
        dmsgs = await _await_messages(collector, 2.0)
        _dump("drone status", dmsgs)

        # --- PROBE A: player-callable POST /drones/<id>/equipment ---
        pa = await rest.post(f"/drones/{drone_id}/equipment", json={"equipment_type": "laser_cannon"})
        print(f"\n[PROBE A /drones/<id>/equipment] http_ok={pa.ok} status={pa.status} data={pa.data}")
        _dump("PROBE A messages", await _await_messages(collector, 3.0))

        if equip_plant is not None:
            plant_id = equip_plant.get("building_id")
            plant_origin = equip_plant.get("origin") or [0, 0]

            # --- PROBE B: manufacture laser at equipment plant (current stock) ---
            pb = await rest.post(
                f"/buildings/{plant_id}/production/manufacture",
                json={"efficiency": 1.0, "equipment_type": "laser_cannon"},
            )
            print(f"\n[PROBE B manufacture] http_ok={pb.ok} status={pb.status} data={pb.data}")
            _dump("PROBE B messages", await _await_messages(collector, 4.0))

            # --- PROBE C: install — try ABSOLUTE drone coords ---
            pc_abs = await rest.post(
                f"/buildings/{plant_id}/equipment_plant/install",
                json={"efficiency": 1.0, "equipment_type": "laser_cannon",
                      "q": drone_loc[0], "r": drone_loc[1]},
            )
            print(f"\n[PROBE C install ABS q,r={drone_loc}] http_ok={pc_abs.ok} status={pc_abs.status} data={pc_abs.data}")
            _dump("PROBE C-abs messages", await _await_messages(collector, 3.0))

            # --- PROBE C2: install — try coords RELATIVE to plant origin ---
            rel_q = drone_loc[0] - plant_origin[0]
            rel_r = drone_loc[1] - plant_origin[1]
            pc_rel = await rest.post(
                f"/buildings/{plant_id}/equipment_plant/install",
                json={"efficiency": 1.0, "equipment_type": "laser_cannon",
                      "q": rel_q, "r": rel_r},
            )
            print(f"\n[PROBE C install REL q,r=({rel_q},{rel_r})] http_ok={pc_rel.ok} status={pc_rel.status} data={pc_rel.data}")
            _dump("PROBE C-rel messages", await _await_messages(collector, 3.0))
        else:
            print("\n!! no equipment plant found; skipping manufacture/install probes")

        # Re-read stored resources to see what manufacture consumed.
        await rest.post("/buildings/command_centre/status_report", json={"efficiency": 1.0})
        msgs2 = await _await_messages(collector, 3.0)
        report2 = next(
            (m for m in msgs2 if m.get("message_type") == "building_action_completed"
             and "status_report" in (m.get("subject") or "").lower()),
            None,
        )
        if report2 is not None:
            picked2 = _pick(report2.get("details") or {})
            for key in ("equipment_plant", "drone_plant", "command_center"):
                b = picked2[key]
                if b is not None:
                    print(f"\n[AFTER {key}] stored={b.get('stored_resources')}")
    finally:
        await socket.stop()
        sock_task.cancel()
        try:
            await sock_task
        except asyncio.CancelledError:
            pass
        await rest.aclose()


if __name__ == "__main__":
    asyncio.run(main())
