"""Hex coordinate utilities for a flat-top odd-q offset grid.

Ported verbatim from the seed client's ``app/hex_utils.py`` (behaviour is
verified against the live game's coordinate system, so we keep it exactly).

Local absolute coordinates: the command center origin (top-left tile) is (0, 0).
The CC origin is always on an even column, so a tile's parity is determined by
whether its absolute ``q`` is even or odd.

All game messages use relative coordinates. This module converts between
relative and absolute coordinates, accounting for column parity.
"""

from __future__ import annotations

# Neighbor offsets for a flat-top odd-q offset grid.
# Index by direction (0=NE, 1=SE, 2=S, 3=SW, 4=NW, 5=N).
EVEN_COL_OFFSETS = [
    (1, -1),   # 0 NE
    (1, 0),    # 1 SE
    (0, 1),    # 2 S
    (-1, 0),   # 3 SW
    (-1, -1),  # 4 NW
    (0, -1),   # 5 N
]

ODD_COL_OFFSETS = [
    (1, 0),    # 0 NE
    (1, 1),    # 1 SE
    (0, 1),    # 2 S
    (-1, 1),   # 3 SW
    (-1, 0),   # 4 NW
    (0, -1),   # 5 N
]


def neighbor_offset(abs_q: int, direction: int) -> tuple[int, int]:
    """Get the (dq, dr) offset for moving in a direction from absolute column abs_q."""
    if abs_q % 2 == 0:
        return EVEN_COL_OFFSETS[direction]
    return ODD_COL_OFFSETS[direction]


def get_neighbor(abs_q: int, abs_r: int, direction: int) -> tuple[int, int]:
    """Get the absolute coordinates of the neighbor in the given direction."""
    dq, dr = neighbor_offset(abs_q, direction)
    return abs_q + dq, abs_r + dr


def rel_to_abs(rel_q: int, rel_r: int, origin_q: int, origin_r: int) -> tuple[int, int]:
    """Convert relative coordinates to local absolute coordinates.

    The game engine computes relative coords by subtracting the origin's
    absolute position, so we reverse that by adding. For status_report coords
    the origin is the CC (0, 0) and this is trivial; for scan results the origin
    is the scanning drone's absolute position.
    """
    return origin_q + rel_q, origin_r + rel_r


def abs_to_rel(abs_q: int, abs_r: int, origin_q: int, origin_r: int) -> tuple[int, int]:
    """Convert local absolute coordinates to relative coordinates for the server.

    The origin is typically the acting drone's absolute position.
    """
    return abs_q - origin_q, abs_r - origin_r


def hex_to_pixel(q: int, r: int, hex_size: float) -> tuple[float, float]:
    """Convert odd-q offset hex coordinates to the pixel center, for flat-top hexes."""
    x = hex_size * 3 / 2 * q
    y = hex_size * 3 ** 0.5 * (r + 0.5 * (q & 1))
    return x, y


def offset_to_cube(q: int, r: int) -> tuple[int, int, int]:
    """Convert odd-q offset coordinates to cube coordinates."""
    x = q
    z = r - (q - (q & 1)) // 2
    y = -x - z
    return x, y, z


def cube_to_offset(x: int, z: int) -> tuple[int, int]:
    """Convert cube coordinates back to odd-q offset coordinates."""
    q = x
    r = z + (x - (x & 1)) // 2
    return q, r


def hex_tiles_in_range(center_q: int, center_r: int, radius: int) -> list[tuple[int, int]]:
    """Return all (q, r) coordinates within hex distance ``radius`` of center (inclusive)."""
    result = [(center_q, center_r)]
    visited = {(center_q, center_r)}
    frontier = [(center_q, center_r)]
    for _ in range(radius):
        next_frontier = []
        for fq, fr in frontier:
            for d in range(6):
                nq, nr = get_neighbor(fq, fr, d)
                if (nq, nr) not in visited:
                    visited.add((nq, nr))
                    result.append((nq, nr))
                    next_frontier.append((nq, nr))
        frontier = next_frontier
    return result


def hex_distance_cube(q1: int, r1: int, q2: int, r2: int) -> int:
    """Hex distance via cube-coordinate conversion."""
    x1, y1, z1 = offset_to_cube(q1, r1)
    x2, y2, z2 = offset_to_cube(q2, r2)
    return max(abs(x1 - x2), abs(y1 - y2), abs(z1 - z2))
