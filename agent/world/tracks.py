"""Enemy tracks: fuse sightings into persistent estimates of enemy assets.

The seed client only ever stored a per-tile enemy *count* from the latest scan —
no identity, no history, no continuity, no decay. This module keeps **tracks**:
each is one enemy we believe exists, updated as new sightings arrive from
different sensors (scan, identify, and — as bearings — seismic / DF antenna),
with a velocity estimate, a predicted position, and confidence that decays with
age so stale tracks fade and expire.

Positional sightings (scan/identify) drive a track's location. Bearing-only
detections (seismic/DF give a direction, not a point) are retained as evidence
that an enemy is active along a heading; turning crossed bearings into points
(triangulation) is left to the sensor-net work (E6.4) — here they raise recency
without moving a track.

Data association is deliberately simple and explicit: by ``drone_id`` when an
identify gives us one, else nearest existing track within a gate distance.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

from agent.bus import EventBus
from agent.rules import hexmath

log = logging.getLogger("agent.world.tracks")

Coord = tuple[int, int]

TOPIC_TRACK_CHANGED = "world.track"

DEFAULT_TRACK_HALF_LIFE = 120     # cycles; enemies move, so decay faster than tiles
DEFAULT_ASSOCIATION_GATE = 3      # hex tiles: a sighting within this joins a track
DEFAULT_MIN_CONFIDENCE = 0.15     # prune below this
DEFAULT_MAX_AGE = 600             # cycles; hard expiry regardless of decay curve
_HISTORY = 8                      # positions kept per track for velocity


class SightingSource(IntEnum):
    SCAN = 1
    IDENTIFY = 2
    SEISMIC = 3      # bearing-only
    DF_ANTENNA = 4   # bearing-only


@dataclass(slots=True)
class Sighting:
    """A positional detection of an enemy at an absolute tile."""

    source: SightingSource
    q: int
    r: int
    cycle: int
    drone_id: str | None = None
    is_decoy: bool = False
    # ``None`` means scan-level contact with an unknown loadout.  Identify is
    # authoritative and supplies a (possibly empty) set.
    equipment: frozenset[str] | None = None

    @property
    def coord(self) -> Coord:
        return (self.q, self.r)


@dataclass(slots=True)
class BearingSighting:
    """A direction-only detection from a passive sensor (seismic / DF antenna).

    ``direction`` is a hex facing 0-5 from the sensor at ``(sensor_q, sensor_r)``.
    Carries no range, so it cannot place a track on its own.
    """

    source: SightingSource
    sensor_q: int
    sensor_r: int
    direction: int
    cycle: int


@dataclass(slots=True)
class EnemyTrack:
    track_id: str
    q: int
    r: int
    last_seen: int
    first_seen: int
    source: SightingSource
    drone_id: str | None = None
    is_decoy: bool = False
    equipment: frozenset[str] = field(default_factory=frozenset)
    equipment_known: bool = False
    half_life: int = DEFAULT_TRACK_HALF_LIFE  # set by the owning TrackStore
    history: deque[tuple[int, int, int]] = field(default_factory=lambda: deque(maxlen=_HISTORY))

    @property
    def coord(self) -> Coord:
        return (self.q, self.r)

    def age(self, now_cycle: int) -> int:
        return max(0, now_cycle - self.last_seen)

    def confidence(self, now_cycle: int) -> float:
        if self.half_life <= 0:
            raise ValueError("half_life must be positive")
        return 0.5 ** (self.age(now_cycle) / self.half_life)

    def velocity(self) -> tuple[float, float]:
        """Estimated (dq, dr) per cycle from the two most recent distinct-cycle
        positions. Naive offset-space extrapolation — good for a short-horizon
        threat lookahead, not long-range prediction. Returns (0, 0) if we have
        fewer than two timestamps or no elapsed time."""
        if len(self.history) < 2:
            return (0.0, 0.0)
        c1, q1, r1 = self.history[-2]
        c2, q2, r2 = self.history[-1]
        dt = c2 - c1
        if dt <= 0:
            return (0.0, 0.0)
        return ((q2 - q1) / dt, (r2 - r1) / dt)

    def predict(self, now_cycle: int) -> Coord:
        """Predicted position at ``now_cycle`` using the velocity estimate."""
        vq, vr = self.velocity()
        dt = now_cycle - self.last_seen
        return (round(self.q + vq * dt), round(self.r + vr * dt))


class TrackStore:
    def __init__(
        self,
        bus: EventBus | None = None,
        *,
        half_life: int = DEFAULT_TRACK_HALF_LIFE,
        gate: int = DEFAULT_ASSOCIATION_GATE,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        max_age: int = DEFAULT_MAX_AGE,
    ) -> None:
        self._bus = bus
        self._half_life = half_life
        self._gate = gate
        self._min_confidence = min_confidence
        self._max_age = max_age
        self._tracks: dict[str, EnemyTrack] = {}
        self._by_drone: dict[str, str] = {}  # drone_id -> track_id
        self._bearings: deque[BearingSighting] = deque(maxlen=64)
        self._counter = 0

    # --- reads ---

    def tracks(self) -> list[EnemyTrack]:
        return list(self._tracks.values())

    def get(self, track_id: str) -> EnemyTrack | None:
        return self._tracks.get(track_id)

    @property
    def count(self) -> int:
        return len(self._tracks)

    def recent_bearings(self) -> list[BearingSighting]:
        return list(self._bearings)

    def nearest(self, coord: Coord) -> EnemyTrack | None:
        best: EnemyTrack | None = None
        best_d = None
        for t in self._tracks.values():
            d = hexmath.hex_distance_cube(coord[0], coord[1], t.q, t.r)
            if best_d is None or d < best_d:
                best, best_d = t, d
        return best

    # --- writes ---

    async def observe(self, s: Sighting) -> EnemyTrack:
        """Fuse a positional sighting into an existing track or create a new one."""
        track = self._associate(s)
        if track is None:
            track = self._create(s)
        else:
            self._update(track, s)
        await self._emit(track.track_id)
        return track

    async def observe_bearing(self, b: BearingSighting) -> None:
        self._bearings.append(b)
        await self._emit(None)

    async def prune(self, now_cycle: int) -> list[str]:
        """Drop tracks that have decayed below the confidence floor or exceeded
        the hard age cap. Returns the removed track ids."""
        removed: list[str] = []
        for tid, t in list(self._tracks.items()):
            if t.age(now_cycle) > self._max_age or t.confidence(now_cycle) < self._min_confidence:
                self._tracks.pop(tid, None)
                if t.drone_id:
                    self._by_drone.pop(t.drone_id, None)
                removed.append(tid)
        for tid in removed:
            await self._emit(tid, removed=True)
        return removed

    # --- association / mutation ---

    def _associate(self, s: Sighting) -> EnemyTrack | None:
        # Strongest signal: a known identity from identify.
        if s.drone_id and s.drone_id in self._by_drone:
            return self._tracks.get(self._by_drone[s.drone_id])
        # Otherwise nearest track within the gate, most-recent as tiebreak.
        best: EnemyTrack | None = None
        best_key = None
        for t in self._tracks.values():
            # Don't merge a positional sighting into a track that already has a
            # *different* confirmed identity.
            if s.drone_id and t.drone_id and t.drone_id != s.drone_id:
                continue
            d = hexmath.hex_distance_cube(s.q, s.r, t.q, t.r)
            if d <= self._gate:
                key = (d, -t.last_seen)
                if best_key is None or key < best_key:
                    best, best_key = t, key
        return best

    def _create(self, s: Sighting) -> EnemyTrack:
        self._counter += 1
        tid = f"trk{self._counter}"
        track = EnemyTrack(
            track_id=tid,
            q=s.q,
            r=s.r,
            last_seen=s.cycle,
            first_seen=s.cycle,
            source=s.source,
            drone_id=s.drone_id,
            is_decoy=s.is_decoy,
            equipment=s.equipment or frozenset(),
            equipment_known=s.equipment is not None,
            half_life=self._half_life,
        )
        track.history.append((s.cycle, s.q, s.r))
        self._tracks[tid] = track
        if s.drone_id:
            self._by_drone[s.drone_id] = tid
        return track

    def _update(self, track: EnemyTrack, s: Sighting) -> None:
        # Newer position wins; ignore an out-of-order (older) sighting for the
        # position/velocity update but still let identity attach.
        if s.cycle >= track.last_seen:
            if (s.q, s.r) != (track.q, track.r) or s.cycle != track.last_seen:
                track.history.append((s.cycle, s.q, s.r))
            track.q, track.r = s.q, s.r
            track.last_seen = s.cycle
            track.source = s.source
        if s.drone_id and track.drone_id is None:
            track.drone_id = s.drone_id
            self._by_drone[s.drone_id] = track.track_id
        # identify is authoritative about decoy status.
        if s.source == SightingSource.IDENTIFY:
            track.is_decoy = s.is_decoy
            if s.equipment is not None:
                track.equipment = s.equipment
                track.equipment_known = True

    async def _emit(self, track_id: str | None, *, removed: bool = False) -> None:
        if self._bus is not None:
            await self._bus.publish(
                TOPIC_TRACK_CHANGED, {"track_id": track_id, "removed": removed}
            )
