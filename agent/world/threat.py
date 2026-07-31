"""Queryable danger costs derived from hostile weapon envelopes.

The pathfinder (E5.1) consumes :class:`ThreatMap` as a callable tile-cost
provider.  Costs are derived on demand from the current world and track stores,
so moving/decaying tracks, identified loadouts, and added or destroyed enemy
buildings are reflected without maintaining a stale second copy of the map.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from agent.rules.hexmath import hex_distance_cube
from agent.world.model import WorldModel
from agent.world.tiles import Coord
from agent.world.tracks import TrackStore


@dataclass(frozen=True, slots=True)
class WeaponEnvelope:
    """A hostile weapon's inclusive range and additive tile cost."""

    min_range: int
    max_range: int
    cost: float = 1.0

    def contains(self, distance: int) -> bool:
        return self.min_range <= distance <= self.max_range


@dataclass(frozen=True, slots=True)
class ThreatSource:
    """One weapon envelope anchored to a currently estimated position."""

    source_id: str
    source_type: str
    coord: Coord
    weapon: str
    envelope: WeaponEnvelope
    confidence: float = 1.0

    def cost_at(self, coord: Coord) -> float:
        distance = hex_distance_cube(self.coord[0], self.coord[1], coord[0], coord[1])
        if not self.envelope.contains(distance):
            return 0.0
        return self.envelope.cost * self.confidence


@dataclass(frozen=True, slots=True)
class ThreatSnapshot:
    """Immutable weapon-source snapshot safe to query from a worker thread."""

    sources: tuple[ThreatSource, ...]

    def cost_at(self, coord: Coord) -> float:
        return sum(source.cost_at(coord) for source in self.sources)

    def costs_at(self, coords: Iterable[Coord]) -> dict[Coord, float]:
        return {coord: self.cost_at(coord) for coord in coords}

    def __call__(self, coord: Coord) -> float:
        return self.cost_at(coord)


# Ticket #17's specified envelopes.  Turret ordering is laser / missile /
# auto-cannon; howitzers are drone-mounted and have a dead zone.
DEFAULT_WEAPON_ENVELOPES: Mapping[str, WeaponEnvelope] = MappingProxyType(
    {
        "laser_turret": WeaponEnvelope(0, 15),
        "missile_turret": WeaponEnvelope(0, 15),
        "auto_cannon_turret": WeaponEnvelope(0, 22),
        "howitzer": WeaponEnvelope(8, 32),
        "laser_cannon": WeaponEnvelope(0, 10),
    }
)


class ThreatMap:
    """Live per-tile threat costs suitable for weighted pathfinding.

    Multiple overlapping weapons add their costs.  Track contributions are
    multiplied by the track's staleness confidence, while identified buildings
    remain certain until a later scan/identify or destruction message removes
    them.  Scan-only contacts use the drone-laser envelope conservatively;
    identify can then prove the contact harmless or reveal a howitzer.
    """

    def __init__(
        self,
        world: WorldModel,
        tracks: TrackStore,
        *,
        weapon_envelopes: Mapping[str, WeaponEnvelope] = DEFAULT_WEAPON_ENVELOPES,
    ) -> None:
        self._world = world
        self._tracks = tracks
        self._weapon_envelopes = dict(weapon_envelopes)

    def sources(self, *, now_cycle: int | None = None) -> tuple[ThreatSource, ...]:
        """Return the weapon envelopes active at ``now_cycle``."""
        cycle = self._world.cycle if now_cycle is None else now_cycle
        result: list[ThreatSource] = []

        for track in self._tracks.tracks():
            if track.is_decoy:
                continue
            track_cycle = max(cycle, track.last_seen)
            weapons = track.equipment & self._weapon_envelopes.keys()
            if not track.equipment_known:
                weapons = {"laser_cannon"}  # conservative scan-only contact
            confidence = track.confidence(track_cycle)
            position = track.predict(track_cycle)
            for weapon in sorted(weapons):
                result.append(
                    ThreatSource(
                        source_id=track.track_id,
                        source_type="drone",
                        coord=position,
                        weapon=weapon,
                        envelope=self._weapon_envelopes[weapon],
                        confidence=confidence,
                    )
                )

        for building in self._world.enemy_buildings():
            weapon = building.building_type
            if weapon not in self._weapon_envelopes or building.origin is None:
                continue
            result.append(
                ThreatSource(
                    source_id=building.building_id,
                    source_type="building",
                    coord=building.origin,
                    weapon=weapon,
                    envelope=self._weapon_envelopes[weapon],
                )
            )
        return tuple(result)

    def cost_at(self, coord: Coord, *, now_cycle: int | None = None) -> float:
        """Return the additive hostile danger cost for one tile."""
        return self.snapshot(now_cycle=now_cycle).cost_at(coord)

    def costs_at(
        self, coords: Iterable[Coord], *, now_cycle: int | None = None
    ) -> dict[Coord, float]:
        """Return costs for several tiles using one source snapshot."""
        return self.snapshot(now_cycle=now_cycle).costs_at(coords)

    def snapshot(self, *, now_cycle: int | None = None) -> ThreatSnapshot:
        """Freeze current sources for a consistent path search or worker task."""

        return ThreatSnapshot(self.sources(now_cycle=now_cycle))

    def __call__(self, coord: Coord) -> float:
        """Allow direct use as a pathfinder tile-cost callback."""
        return self.cost_at(coord)
