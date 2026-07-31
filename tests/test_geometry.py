"""Tests for elevation-aware range, LOS, and wire coordinate frames."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.rules.geometry import (
    CoordinateFrame,
    absolute_to_relative,
    distance_3d,
    has_line_of_sight,
    hex_line,
    relative_to_absolute,
    within_3d_range,
)

GOLDEN_SAMPLES = Path(__file__).parent / "fixtures" / "message_samples.json"


def test_distance_3d_scales_five_altitude_units_as_one_tile() -> None:
    # Four horizontal tiles and 15 altitude units form a 3-4-5 triangle.
    assert distance_3d((0, 0), (0, 4), target_elevation=15) == pytest.approx(5.0)
    assert distance_3d(
        (0, 0),
        (0, 4),
        origin_elevation=20,
        target_elevation=5,
    ) == pytest.approx(5.0)


def test_elevation_can_put_a_horizontally_close_shot_out_of_range() -> None:
    origin = (0, 0)
    target = (0, 6)

    assert within_3d_range(origin, target, 6)
    assert not within_3d_range(origin, target, 9, target_elevation=40)
    assert within_3d_range(origin, target, 10, target_elevation=40)


def test_negative_max_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        within_3d_range((0, 0), (0, 0), -1)


def test_hex_line_includes_each_endpoint_and_one_tile_per_step() -> None:
    assert hex_line((0, 0), (0, 4)) == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    assert hex_line((-2, 3), (-2, 3)) == [(-2, 3)]

    ambiguous_line = hex_line((-7, 3), (8, -5))
    assert hex_line((8, -5), (-7, 3)) == list(reversed(ambiguous_line))


def test_ridge_blocks_line_of_sight() -> None:
    terrain = {
        (0, 0): 0,
        (0, 1): 0,
        (0, 2): 8,
        (0, 3): 0,
        (0, 4): 0,
    }

    assert not has_line_of_sight((0, 0), (0, 4), terrain)
    assert has_line_of_sight(
        (0, 0),
        (0, 4),
        terrain,
        origin_elevation=10,
        target_elevation=10,
    )


def test_los_interpolates_between_different_endpoint_heights() -> None:
    terrain = {(0, step): 0 for step in range(5)}
    terrain[(0, 2)] = 9

    # At the midpoint, a ray from altitude 0 to altitude 20 is at height 10.
    assert has_line_of_sight((0, 0), (0, 4), terrain, target_elevation=20)

    terrain[(0, 2)] = 11
    assert not has_line_of_sight((0, 0), (0, 4), terrain, target_elevation=20)


def test_unknown_terrain_is_conservatively_blocking() -> None:
    terrain = {(0, 0): 0, (0, 1): 0, (0, 3): 0}

    assert not has_line_of_sight((0, 0), (0, 3), terrain)
    assert has_line_of_sight((0, 0), (0, 3), terrain, unknown_is_blocking=False)


def test_coordinate_frame_converts_scan_wire_samples_using_actor_origin() -> None:
    samples = json.loads(GOLDEN_SAMPLES.read_text(encoding="utf-8"))
    scan = next(payload for payload in samples if payload["subject"] == "Scan completed")
    relative_tiles = [tuple(tile["coordinates"]) for tile in scan["details"]["scan_data"]["tiles"]]
    assert (-1, -1) in relative_tiles
    assert (5, 0) in relative_tiles

    frame = CoordinateFrame(origin=(7, -3))
    absolute_tiles = [frame.to_absolute(coordinate) for coordinate in relative_tiles]

    assert (6, -4) in absolute_tiles
    assert (12, -3) in absolute_tiles
    assert [frame.to_relative(coordinate) for coordinate in absolute_tiles] == relative_tiles


def test_free_coordinate_helpers_require_an_explicit_origin() -> None:
    assert relative_to_absolute((1, -1), origin=(-4, 9)) == (-3, 8)
    assert absolute_to_relative((-3, 8), origin=(-4, 9)) == (1, -1)
