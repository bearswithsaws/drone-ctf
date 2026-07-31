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
from agent.world.tracks import Sighting, SightingSource, TrackStore

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

    def __init__(self, world: WorldModel, tracks: "TrackStore | None" = None) -> None:
        self._world = world
        self._tracks = tracks
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
        enemy_coords: list[tuple[int, int]] = []
        empty_building_coords: list[tuple[int, int]] = []
        for tile in scan_data.get("tiles", []) or []:
            rel = _coord(tile.get("coordinates"))
            if rel is None:
                continue
            aq, ar = relative_to_absolute(rel, origin=origin)
            has_drone = bool(tile.get("has_drone", False))
            obs.append(
                TileObservation(
                    q=aq,
                    r=ar,
                    terrain_type=tile.get("terrain_type"),
                    elevation=tile.get("elevation"),
                    has_resource=bool(tile.get("has_resource", False)),
                    has_building=bool(tile.get("has_building", False)),
                    has_drone=has_drone,
                )
            )
            # A scanned drone that isn't one of ours is an enemy sighting. Scan
            # gives no identity, so this is a position-only track seed.
            if has_drone and not self._own_occupied((aq, ar)):
                enemy_coords.append((aq, ar))
            if not tile.get("has_building", False):
                empty_building_coords.append((aq, ar))
        if obs:
            await self._world.observe_tiles(obs, cycle=cycle, source=Source.SCAN)
        for coord in empty_building_coords:
            await self._world.remove_enemy_buildings_at(coord)
        if self._tracks is not None:
            for aq, ar in enemy_coords:
                await self._tracks.observe(Sighting(SightingSource.SCAN, aq, ar, cycle))

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
        building = data.get("building")
        building_id = _building_id(building, (aq, ar))
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
                has_building=bool(building),
                building_id=building_id,
            ),
            cycle=cycle,
            source=Source.IDENTIFY,
        )
        if building and building_id is not None:
            # An identify of our own footprint can include friendly detail; only
            # the other buildings belong in the hostile inventory.
            if self._world.get_building(building_id) is None:
                await self._world.upsert_enemy_building(
                    building_id,
                    building_type=_building_type(building),
                    origin=(aq, ar),
                    tiles=((aq, ar),),
                    cycle=cycle,
                )
        else:
            await self._world.remove_enemy_buildings_at((aq, ar))
        # Identify resolves enemy identity + decoy status at this tile.
        if self._tracks is not None:
            for d in data.get("drones", []) or []:
                did = d.get("drone_id")
                if did and self._world.get_drone(did) is not None:
                    continue  # one of ours
                # Seed heuristic (also used by the reference client): a 1-HP unit
                # with <=1 equipment is a decoy.
                is_decoy = d.get("max_health") == 1 and len(d.get("equipment") or []) <= 1
                await self._tracks.observe(
                    Sighting(
                        SightingSource.IDENTIFY,
                        aq,
                        ar,
                        cycle,
                        drone_id=did,
                        is_decoy=is_decoy,
                        equipment=_equipment_types(d.get("equipment") or []),
                    )
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
            await self._world.remove_enemy_building(building_id)

    # --- helpers ---

    def _drone_origin(self, drone_id: str) -> tuple[int, int] | None:
        rec = self._world.get_drone(drone_id)
        if rec is None:
            return None
        return rec.coord

    def _own_occupied(self, coord: tuple[int, int]) -> bool:
        """Whether one of our own drones is at ``coord`` (so a scanned drone
        there is friendly, not an enemy contact)."""
        return any(d.coord == coord for d in self._world.drones())

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


def _building_id(building: Any, coord: tuple[int, int]) -> str | None:
    if not isinstance(building, dict):
        return None
    value = building.get("building_id") or building.get("id")
    return str(value) if value else f"observed:{coord[0]}:{coord[1]}"


def _building_type(building: Any) -> str | None:
    if not isinstance(building, dict):
        return None
    value = building.get("building_type") or building.get("type")
    return _normalise_equipment_name(value)


def _equipment_types(equipment: list[Any]) -> frozenset[str]:
    result: set[str] = set()
    for item in equipment:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = (
                item.get("equipment_type")
                or item.get("type")
                or item.get("name")
            )
        else:
            continue
        normalised = _normalise_equipment_name(name)
        if normalised:
            result.add(normalised)
    return frozenset(result)


def _normalise_equipment_name(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower().replace("-", "_").replace(" ", "_")
