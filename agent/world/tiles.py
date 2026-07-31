"""Tile beliefs for the world model.

A :class:`TileBelief` is what we *think* is true about one hex, tagged with the
cycle it was last observed and the observation source. Confidence decays with
age (in engine cycles) so downstream planners can prefer fresh knowledge and
re-scan stale frontiers. This module is pure — no I/O, no bus, no message
parsing (that mapping lives in the E3.2 ingest layer).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

Coord = tuple[int, int]

# Default confidence half-life in engine cycles. At ~0.25s/cycle this is ~50s;
# tunable per WorldModel. A tile observed `half_life` cycles ago has confidence
# 0.5, two half-lives ago 0.25, etc.
DEFAULT_CONFIDENCE_HALF_LIFE = 200


class Terrain(IntEnum):
    """Server terrain_type codes."""

    NORMAL = 1      # passable, 1x movement cost
    DIFFICULT = 2   # passable, 2x ground-movement cost (hover ignores)
    IMPASSABLE = 3  # cannot be entered on the ground


class Source(IntEnum):
    """How a tile belief was last established.

    Ordering lets a later ingest decide whether a new observation should
    supersede detail from an older one (identify carries more than a scan; a
    status_report is authoritative about our own tiles; admin is ground truth).
    """

    SCAN = 1
    IDENTIFY = 2
    STATUS_REPORT = 3
    ADMIN = 4  # ground-truth from the admin oracle (calibration only)


@dataclass(slots=True)
class TileBelief:
    q: int
    r: int
    terrain_type: int | None = None
    elevation: int | None = None
    has_resource: bool = False
    resource_type: str | None = None
    resource_amount: int | None = None
    has_building: bool = False
    building_id: str | None = None
    # Occupancy is a scan-level hint (counts only); enemy identity lives in
    # tracks (E3.3) and own-entity records on the model.
    has_drone: bool = False
    last_seen: int = 0
    source: Source = Source.SCAN

    @property
    def coord(self) -> Coord:
        return (self.q, self.r)

    @property
    def passable(self) -> bool:
        """Whether the ground is enterable. Unknown terrain is treated as
        passable (optimistic) so unexplored tiles don't block pathing outright —
        the unknown-tile penalty in pathfinding handles caution."""
        return self.terrain_type != Terrain.IMPASSABLE

    def age(self, now_cycle: int) -> int:
        """Cycles elapsed since this tile was last observed (never negative)."""
        return max(0, now_cycle - self.last_seen)

    def confidence(
        self, now_cycle: int, half_life: int = DEFAULT_CONFIDENCE_HALF_LIFE
    ) -> float:
        """Exponential-decay confidence in [0, 1]: 1.0 when just seen, 0.5 after
        one half-life, etc. ``half_life`` must be positive."""
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        return 0.5 ** (self.age(now_cycle) / half_life)
