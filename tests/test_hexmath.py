"""Property + unit tests for the odd-q hex math port."""

from __future__ import annotations

import itertools

from agent.rules import hexmath as hx

# A representative spread of coordinates including both column parities and
# negatives (the map extends in all directions from the CC origin).
COORDS = list(itertools.product(range(-6, 7), range(-6, 7)))


def test_offset_cube_roundtrip() -> None:
    for q, r in COORDS:
        x, _y, z = hx.offset_to_cube(q, r)
        assert hx.cube_to_offset(x, z) == (q, r)


def test_cube_invariant() -> None:
    # Cube coords must always sum to zero.
    for q, r in COORDS:
        x, y, z = hx.offset_to_cube(q, r)
        assert x + y + z == 0


def test_distance_symmetry() -> None:
    for (q1, r1), (q2, r2) in itertools.product(COORDS[::7], COORDS[::5]):
        assert hx.hex_distance_cube(q1, r1, q2, r2) == hx.hex_distance_cube(q2, r2, q1, r1)


def test_distance_self_is_zero() -> None:
    for q, r in COORDS:
        assert hx.hex_distance_cube(q, r, q, r) == 0


def test_neighbors_are_distance_one() -> None:
    for q, r in COORDS:
        for d in range(6):
            nq, nr = hx.get_neighbor(q, r, d)
            assert hx.hex_distance_cube(q, r, nq, nr) == 1


def test_neighbor_closure_six_distinct() -> None:
    # Each tile has exactly six distinct neighbors.
    for q, r in COORDS:
        neigh = {hx.get_neighbor(q, r, d) for d in range(6)}
        assert len(neigh) == 6
        assert (q, r) not in neigh


def test_range_counts_match_hex_formula() -> None:
    # A filled hex of radius R contains 3R(R+1)+1 tiles.
    for radius in range(0, 6):
        tiles = hx.hex_tiles_in_range(0, 0, radius)
        assert len(tiles) == len(set(tiles))  # no duplicates
        assert len(tiles) == 3 * radius * (radius + 1) + 1


def test_range_membership_matches_distance() -> None:
    radius = 4
    tiles = set(hx.hex_tiles_in_range(2, -1, radius))
    for q, r in COORDS:
        in_range = hx.hex_distance_cube(2, -1, q, r) <= radius
        assert ((q, r) in tiles) == in_range


def test_rel_abs_roundtrip() -> None:
    origin = (3, -2)
    for q, r in COORDS:
        rel = hx.abs_to_rel(q, r, *origin)
        assert hx.rel_to_abs(*rel, *origin) == (q, r)


def test_opposite_directions_return_to_origin() -> None:
    # NE(0)/SW(3), SE(1)/NW(4), S(2)/N(5) are opposite pairs.
    for q, r in COORDS:
        for d, opp in ((0, 3), (1, 4), (2, 5)):
            nq, nr = hx.get_neighbor(q, r, d)
            assert hx.get_neighbor(nq, nr, opp) == (q, r)
