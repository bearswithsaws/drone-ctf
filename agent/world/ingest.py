"""Ingest registry: wire messages -> world-model updates.

Replaces the seed client's ~130-line ``elif`` dispatcher (``message_parser.py``)
with a small ordered rule table. Each inbound message (published on the bus
under ``agent.transport.ws.TOPIC_MESSAGE``) is matched by ``(message_type,
subject-substring)`` to one handler that translates it into
:class:`~agent.world.model.WorldModel` mutations.

Coordinate handling (see ``agent.rules.geometry``):
- **status_report** coords are already relative to our CC origin = absolute
  ``(0, 0)`` — no conversion.
- **scan / identify** coords are relative to the acting drone; we convert with
  the drone's known absolute position.

Improvement over the seed client: scans that arrive before we know the acting
drone's position are **buffered and replayed** once the position is learned,
instead of being silently discarded (the seed client's
``update_from_scan`` early-returns and drops them).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from agent.rules import hexmath
from agent.rules.geometry import relative_to_absolute
from agent.world.model import TileObservation, WorldModel
from agent.world.tiles import Source

log = logging.getLogger("agent.world.ingest")

Handler = Callable[["Ingestor", dict[str, Any], int], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class Rule:
    message_type: str
    subject_contains: str  # lowercased substring; "" matches any subject
    handler: Handler

    def matches(self, message_type: str, subject: str) -> bool:
        return self.message_type == message_type and self.subject_contains in subject


class Ingestor:
    """Applies inbound messages to a :class:`WorldModel`. Subscribe :meth:`on_message`
    to the bus's message topic."""

    def __init__(self, world: WorldModel) -> None:
        self._world = world
        # drone_id -> list of (scan_data, cycle) awaiting a known drone position.
        self._pending_scans: dict[str, list[tuple[dict[str, Any], int]]] = {}
        self._rules = self._build_rules()

    # --- bus entry point ---

    async def on_message(self, _topic: str, msg: Any) -> None:
        if isinstance(msg, dict):
            await self.ingest(msg)

    async def ingest(self, msg: dict[str, Any]) -> None:
        message_type = msg.get("message_type", "") or ""
        subject = (msg.get("subject") or "").lower()
        cycle = msg.get("timestamp") or 0
        for rule in self._rules:
            if rule.matches(message_type, subject):
                await rule.handler(self, msg, cycle)
                return
        # Unmapped types are expected (many message types don't touch the map).
        # They're preserved by the telemetry recorder; here we only trace them.
        log.debug("no ingest rule for message_type=%s subject=%r", message_type, subject)

    # --- rule table (order matters: most specific first) ---

    def _build_rules(self) -> list[Rule]:
        return [
            Rule("building_action_completed", "status_report", Ingestor._on_status_report),
            Rule("building_action_completed", "status completed", Ingestor._on_building_status),
            Rule("action_completed", "scan", Ingestor._on_scan),
            Rule("action_completed", "identify", Ingestor._on_identify),
            Rule("action_completed", "drive", Ingestor._on_drive),
            Rule("action_completed", "turn", Ingestor._on_turn),
            Rule("action_completed", "status completed", Ingestor._on_drone_status),
            Rule("drone_destroyed", "", Ingestor._on_drone_destroyed),
            Rule("building_destroyed", "", Ingestor._on_building_destroyed),
        ]

    # --- handlers ---

    async def _on_status_report(self, msg: dict[str, Any], cycle: int) -> None:
        """Authoritative sweep: coords are CC-relative == absolute. Anything not in
        the report is removed; drone/building positions are canonical."""
        details = msg.get("details") or {}
        buildings = details.get("buildings", []) or []
        drones = details.get("drones", []) or []

        report_bids = {b.get("building_id") for b in buildings if b.get("building_id")}
        for bid in [b.building_id for b in self._world.buildings() if b.building_id not in report_bids]:
            await self._world.remove_building(bid)

        report_dids = {d.get("drone_id") for d in drones if d.get("drone_id")}
        for did in [d.drone_id for d in self._world.drones() if d.drone_id not in report_dids]:
            await self._world.remove_drone(did)

        for bld in buildings:
            bid = bld.get("building_id")
            if not bid:
                continue
            origin = _coord(bld.get("origin"))
            tiles = [_coord(c) for c in bld.get("coordinates", []) if _coord(c) is not None]
            await self._world.upsert_building(
                bid,
                building_type=bld.get("building_type"),
                origin=origin,
                tiles=[t for t in tiles if t is not None],
                cycle=cycle,
            )
            # Buildings occupy their footprint tiles.
            obs = [
                TileObservation(q=t[0], r=t[1], has_building=True, building_id=bid)
                for t in tiles
                if t is not None
            ]
            if obs:
                await self._world.observe_tiles(obs, cycle=cycle, source=Source.STATUS_REPORT)

        for drn in drones:
            did = drn.get("drone_id")
            loc = _coord(drn.get("location"))
            if not did or loc is None:
                continue
            await self._world.upsert_drone(did, q=loc[0], r=loc[1], cycle=cycle)
            await self._flush_pending_scans(did, cycle)

    async def _on_scan(self, msg: dict[str, Any], cycle: int) -> None:
        drone_id = msg.get("drone_id")
        details = msg.get("details") or {}
        scan_data = details.get("scan_data") or {}
        if not drone_id or not scan_data:
            return
        origin = self._drone_origin(drone_id)
        if origin is None:
            # Position unknown — buffer and replay once we learn it (seed-bug fix).
            self._pending_scans.setdefault(drone_id, []).append((scan_data, cycle))
            log.debug("buffered scan for drone %s (position unknown)", drone_id)
            return
        await self._apply_scan(scan_data, origin, cycle)

    async def _apply_scan(self, scan_data: dict[str, Any], origin: tuple[int, int], cycle: int) -> None:
        obs: list[TileObservation] = []
        for tile in scan_data.get("tiles", []) or []:
            rel = _coord(tile.get("coordinates"))
            if rel is None:
                continue
            aq, ar = relative_to_absolute(rel, origin=origin)
            obs.append(
                TileObservation(
                    q=aq,
                    r=ar,
                    terrain_type=tile.get("terrain_type"),
                    elevation=tile.get("elevation"),
                    has_resource=bool(tile.get("has_resource", False)),
                    has_building=bool(tile.get("has_building", False)),
                    has_drone=bool(tile.get("has_drone", False)),
                )
            )
        if obs:
            await self._world.observe_tiles(obs, cycle=cycle, source=Source.SCAN)

    async def _on_identify(self, msg: dict[str, Any], cycle: int) -> None:
        drone_id = msg.get("drone_id")
        details = msg.get("details") or {}
        data = details.get("identify_data") or {}
        if not drone_id or not data:
            return
        origin = self._drone_origin(drone_id)
        if origin is None:
            log.debug("identify for drone %s dropped (position unknown)", drone_id)
            return
        rel = (int(data.get("target_q", 0)), int(data.get("target_r", 0)))
        aq, ar = relative_to_absolute(rel, origin=origin)
        resource = data.get("resource") or {}
        has_resource = bool(
            resource
            and not resource.get("depleted", False)
            and (resource.get("ore_volume", 0) or 0) > 0
        )
        await self._world.observe_tile(
            TileObservation(
                q=aq,
                r=ar,
                terrain_type=data.get("terrain"),
                elevation=data.get("elevation"),
                has_resource=has_resource,
                resource_type=resource.get("ore_type") if has_resource else None,
                resource_amount=resource.get("ore_volume") if has_resource else None,
            ),
            cycle=cycle,
            source=Source.IDENTIFY,
        )

    async def _on_drive(self, msg: dict[str, Any], cycle: int) -> None:
        drone_id = msg.get("drone_id")
        details = msg.get("details") or {}
        if not drone_id or not details.get("success"):
            return
        rec = self._world.get_drone(drone_id)
        if rec is None or rec.coord is None or rec.direction is None:
            return
        direction = rec.direction
        if details.get("drive_direction") == "reverse":
            direction = (direction + 3) % 6
        nq, nr = hexmath.get_neighbor(rec.q, rec.r, direction)
        await self._world.upsert_drone(drone_id, q=nq, r=nr, cycle=cycle)

    async def _on_turn(self, msg: dict[str, Any], cycle: int) -> None:
        drone_id = msg.get("drone_id")
        details = msg.get("details") or {}
        if not drone_id or not details.get("success"):
            return
        new_direction = details.get("new_direction")
        if new_direction is None:
            return
        await self._world.upsert_drone(drone_id, direction=int(new_direction), cycle=cycle)

    async def _on_drone_status(self, msg: dict[str, Any], cycle: int) -> None:
        """Drone status carries no position, but does carry heading/elevation."""
        drone_id = msg.get("drone_id")
        status = (msg.get("details") or {}).get("status") or {}
        if not drone_id:
            return
        await self._world.upsert_drone(
            drone_id,
            direction=status.get("direction"),
            elevation=status.get("elevation"),
            cycle=cycle,
        )

    async def _on_building_status(self, msg: dict[str, Any], cycle: int) -> None:
        building_id = msg.get("building_id")
        status = (msg.get("details") or {}).get("status") or {}
        if not building_id:
            return
        # Individual building status returns self-relative origin (0,0); do NOT
        # overwrite the building's real origin (learned from status_report).
        await self._world.upsert_building(
            building_id, building_type=status.get("building_type"), cycle=cycle
        )

    async def _on_drone_destroyed(self, msg: dict[str, Any], cycle: int) -> None:
        drone_id = msg.get("drone_id")
        if drone_id:
            await self._world.remove_drone(drone_id)

    async def _on_building_destroyed(self, msg: dict[str, Any], cycle: int) -> None:
        building_id = msg.get("building_id")
        if building_id:
            await self._world.remove_building(building_id)

    # --- helpers ---

    def _drone_origin(self, drone_id: str) -> tuple[int, int] | None:
        rec = self._world.get_drone(drone_id)
        if rec is None:
            return None
        return rec.coord

    async def _flush_pending_scans(self, drone_id: str, cycle: int) -> None:
        """Replay scans buffered before the drone's position was known.

        Only called from status_report (the canonical position source). Assumes
        the drone was stationary between the buffered scan and this first fix —
        true at match start, which is exactly when scans-before-first-report
        would otherwise be lost. Not called after a drive (which moves the drone
        and would invalidate a pre-move scan's origin)."""
        pending = self._pending_scans.pop(drone_id, None)
        if not pending:
            return
        origin = self._drone_origin(drone_id)
        if origin is None:
            self._pending_scans[drone_id] = pending  # still unknown; keep buffered
            return
        log.debug("replaying %d buffered scan(s) for drone %s", len(pending), drone_id)
        for scan_data, scan_cycle in pending:
            await self._apply_scan(scan_data, origin, scan_cycle)


def _coord(value: Any) -> tuple[int, int] | None:
    """Coerce a [q, r] / (q, r) wire value into an int tuple, or None."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return None
    return None
