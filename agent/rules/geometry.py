"""Elevation-aware geometry for the odd-q game map.

Horizontal distances are measured in hex tiles.  Elevation is measured in the
server's altitude units, where five altitude units are equivalent to one tile.
Wire coordinates are relative to the entity performing an action; a
``CoordinateFrame`` binds those conversions to that entity's absolute origin.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import hypot

from agent.rules import hexmath

HexCoordinate = tuple[int, int]
ElevationMap = Mapping[HexCoordinate, float]

ALTITUDE_UNITS_PER_TILE = 5.0
_LOS_EPSILON = 1e-9


def relative_to_absolute(
    relative: HexCoordinate,
    *,
    origin: HexCoordinate,
) -> HexCoordinate:
    """Convert a wire-relative coordinate using the acting entity's origin."""
    return hexmath.rel_to_abs(*relative, *origin)


def absolute_to_relative(
    absolute: HexCoordinate,
    *,
    origin: HexCoordinate,
) -> HexCoordinate:
    """Convert an absolute coordinate for an action issued by ``origin``."""
    return hexmath.abs_to_rel(*absolute, *origin)


@dataclass(frozen=True, slots=True)
class CoordinateFrame:
    """Relative/absolute conversions bound to one acting entity's origin."""

    origin: HexCoordinate

    def to_absolute(self, relative: HexCoordinate) -> HexCoordinate:
        """Convert a coordinate received from the server into world space."""
        return relative_to_absolute(relative, origin=self.origin)

    def to_relative(self, absolute: HexCoordinate) -> HexCoordinate:
        """Convert a world coordinate into an action's wire coordinate."""
        return absolute_to_relative(absolute, origin=self.origin)


def distance_3d(
    origin: HexCoordinate,
    target: HexCoordinate,
    *,
    origin_elevation: float = 0,
    target_elevation: float = 0,
) -> float:
    """Return effective range in tiles, accounting for elevation difference."""
    horizontal = hexmath.hex_distance_cube(*origin, *target)
    vertical = abs(target_elevation - origin_elevation) / ALTITUDE_UNITS_PER_TILE
    return hypot(horizontal, vertical)


def within_3d_range(
    origin: HexCoordinate,
    target: HexCoordinate,
    max_range: float,
    *,
    origin_elevation: float = 0,
    target_elevation: float = 0,
) -> bool:
    """Return whether ``target`` is within the inclusive effective range."""
    if max_range < 0:
        raise ValueError("max_range must be non-negative")
    return (
        distance_3d(
            origin,
            target,
            origin_elevation=origin_elevation,
            target_elevation=target_elevation,
        )
        <= max_range
    )


def hex_line(origin: HexCoordinate, target: HexCoordinate) -> list[HexCoordinate]:
    """Return every tile on a straight hex ray, including both endpoints."""
    # Canonicalise direction so an edge-crossing tie selects the same tiles in
    # both directions.  This also makes LOS symmetric when endpoint heights are
    # swapped.
    if origin > target:
        return list(reversed(hex_line(target, origin)))

    distance = hexmath.hex_distance_cube(*origin, *target)
    if distance == 0:
        return [origin]

    start = _nudged_cube(origin)
    end = _nudged_cube(target)
    result: list[HexCoordinate] = []
    for step in range(distance + 1):
        fraction = step / distance
        cube = tuple(start[axis] + (end[axis] - start[axis]) * fraction for axis in range(3))
        x, _y, z = _round_cube(*cube)
        result.append(hexmath.cube_to_offset(x, z))
    return result


def has_line_of_sight(
    origin: HexCoordinate,
    target: HexCoordinate,
    terrain_elevations: ElevationMap,
    *,
    origin_elevation: float = 0,
    target_elevation: float = 0,
    unknown_is_blocking: bool = True,
) -> bool:
    """Return whether terrain leaves an unobstructed sight line.

    ``origin_elevation`` and ``target_elevation`` are entity height above the
    terrain at their respective tiles (for example, drone hover or antenna
    height).  Intermediate terrain strictly above the interpolated ray blocks
    sight.  Missing terrain is conservatively blocking by default; callers may
    opt out when planning through unknown map tiles.

    This function models terrain elevation only.  Buildings, resources, smoke,
    and weapon-specific firing restrictions should be handled by their own
    rules before or after this check.
    """
    line = hex_line(origin, target)
    if len(line) == 1:
        return True

    origin_terrain = _known_elevation(
        origin, terrain_elevations, unknown_is_blocking=unknown_is_blocking
    )
    target_terrain = _known_elevation(
        target, terrain_elevations, unknown_is_blocking=unknown_is_blocking
    )
    if origin_terrain is None or target_terrain is None:
        return False

    sight_origin = origin_terrain + origin_elevation
    sight_target = target_terrain + target_elevation
    segments = len(line) - 1

    for step, coordinate in enumerate(line[1:-1], start=1):
        terrain = _known_elevation(
            coordinate,
            terrain_elevations,
            unknown_is_blocking=unknown_is_blocking,
        )
        if terrain is None:
            return False
        ray_height = sight_origin + (sight_target - sight_origin) * step / segments
        if terrain > ray_height + _LOS_EPSILON:
            return False
    return True


def _known_elevation(
    coordinate: HexCoordinate,
    terrain_elevations: ElevationMap,
    *,
    unknown_is_blocking: bool,
) -> float | None:
    try:
        return float(terrain_elevations[coordinate])
    except KeyError:
        return None if unknown_is_blocking else 0.0


def _nudged_cube(coordinate: HexCoordinate) -> tuple[float, float, float]:
    """Nudge a cube point so rays crossing an exact edge are deterministic."""
    x, y, z = hexmath.offset_to_cube(*coordinate)
    return x + 1e-6, y + 1e-6, z - 2e-6


def _round_cube(x: float, y: float, z: float) -> tuple[int, int, int]:
    rounded_x, rounded_y, rounded_z = round(x), round(y), round(z)
    delta_x = abs(rounded_x - x)
    delta_y = abs(rounded_y - y)
    delta_z = abs(rounded_z - z)

    if delta_x > delta_y and delta_x > delta_z:
        rounded_x = -rounded_y - rounded_z
    elif delta_y > delta_z:
        rounded_y = -rounded_x - rounded_z
    else:
        rounded_z = -rounded_x - rounded_y
    return rounded_x, rounded_y, rounded_z
