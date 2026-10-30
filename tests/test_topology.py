import json

import numpy as np
import pytest
import shapely

from geosimplify import Layer, Simplifier, resolution, to_mercator
from geosimplify.core import crossing_segments
from geosimplify.encoding import _zigzag, encode_feature, layer_size, quantize
from geosimplify.metrics import coverage_errors, layer_metrics, to_shapely, vertex_count
from geosimplify.topology import ArcGraph, find_junctions


def jagged(x0, x1, y, n, amp, seed):
    """A noisy horizontal border from (x0, y) to (x1, y)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(x0, x1, n)
    yy = y + rng.uniform(-amp, amp, n)
    yy[0] = yy[-1] = y
    return np.column_stack([x, yy])


@pytest.fixture
def two_countries():
    """Two polygons sharing a jagged border, plus an island; coordinates in metres."""
    border = jagged(0, 1000, 500, 200, 40, 1)
    south = np.vstack([[[0, 0], [1000, 0]], border[::-1], [[0, 0]]])
    north = np.vstack([border, [[1000, 1000], [0, 1000]], border[:1]])
    t = np.linspace(0, 2 * np.pi, 60)[:-1]
    island = np.column_stack([1500 + 80 * np.cos(t), 500 + 80 * np.sin(t)])
    island = np.vstack([island, island[:1]])
    return Layer("polygon", [[[south]], [[north]], [[island]]], [{}, {}, {}])


def test_junctions_and_arcs(two_countries):
    g = ArcGraph.build("polygon", two_countries.features, topology=True)
    # the shared border is stored once: 2 border endpoints are junctions
    assert len(find_junctions([r for f in two_countries.features for p in f for r in p])) == 2
    assert len(g.arcs) == 4  # border, south remainder, north remainder, island
    shared = [i for i in range(len(g.arcs)) if sum(i in {a for a, _ in seq} for _, seq in g.paths) == 2]
    assert len(shared) == 1


def test_topology_rebuild_is_lossless(two_countries):
    g = ArcGraph.build("polygon", two_countries.features, topology=True)
    rebuilt = g.assemble([a.coords for a in g.arcs])
    a0 = shapely.area(to_shapely("polygon", two_countries.features))
    a1 = shapely.area(to_shapely("polygon", rebuilt))
    np.testing.assert_allclose(a0, a1)


def test_topology_mode_has_no_slivers_independent_does(two_countries):
    orig_union = shapely.union_all(to_shapely("polygon", two_countries.features))
    errs = {}
    for mode in ("independent", "topology"):
        out = Simplifier(two_countries, "dp", mode).simplify(30.0)
        errs[mode] = coverage_errors(orig_union, to_shapely("polygon", out))
    assert errs["topology"]["overlap_area"] == pytest.approx(0, abs=1e-6)
    assert errs["topology"]["gap_count"] == 0
    assert errs["independent"]["overlap_area"] > 0 or errs["independent"]["gap_count"] > 0


def test_closed_ring_never_collapses_to_a_line(two_countries):
    out = Simplifier(two_countries, "dp", "topology").simplify(10_000.0)
    island = out[2]
    assert island and len(island[0][0]) >= 4


def test_safe_mode_removes_crossings():
    # A's top edge has a shallow notch that B sits in; plain DP straightens the
    # notch and the new edge cuts through B. Safe mode restores the notch.
    a = np.array([[0, 0], [100, 0], [100, 100], [60, 100], [60, 90], [40, 90], [40, 100], [0, 100], [0, 0]], float)
    b = np.array([[45, 95], [55, 95], [55, 110], [45, 110], [45, 95]], float)
    layer = Layer("polygon", [[[a]], [[b]]], [{}, {}])
    plain = Simplifier(layer, "dp", "topology")
    plain.simplify(15.0)
    assert crossing_segments(plain.graph)
    safe = Simplifier(layer, "dp", "safe")
    out = safe.simplify(15.0)
    assert crossing_segments(safe.graph) == [] and safe.remaining_crossings == 0
    assert shapely.is_valid(to_shapely("polygon", out)).all()
    assert to_shapely("polygon", out)[0].intersection(to_shapely("polygon", out)[1]).area == 0


def test_safe_mode_keeps_island_outside_coast():
    # B sits in a shallow bay of A without touching it. Straightening the bay
    # swallows B (no segments cross); the side check must restore the bay.
    a = np.array([[0, 0], [100, 0], [100, 100], [60, 100], [60, 92], [40, 92], [40, 100], [0, 100], [0, 0]], float)
    b = np.array([[48, 94], [52, 94], [52, 98], [48, 98], [48, 94]], float)
    layer = Layer("polygon", [[[a]], [[b]]], [{}, {}])
    plain = Simplifier(layer, "dp", "topology")
    g = to_shapely("polygon", plain.simplify(10.0))
    assert g[0].contains(g[1]) and not crossing_segments(plain.graph)
    safe = Simplifier(layer, "dp", "safe")
    g = to_shapely("polygon", safe.simplify(10.0))
    assert not g[0].intersects(g[1])
    assert safe.remaining_side_conflicts == 0


def test_tiny_triangle_ring_survives():
    tri = np.array([[0, 0], [1, 0], [0, 1], [0, 0]], float)
    layer = Layer("polygon", [[[tri]]], [{}])
    for method in ("radial", "dp", "vw"):
        out = Simplifier(layer, method, "independent").simplify(100.0)
        assert out[0] and len(out[0][0][0]) == 4


def test_metrics_and_displacement(two_countries):
    s = Simplifier(two_countries, "dp", "topology")
    out = s.simplify(20.0)
    orig = to_shapely("polygon", two_countries.features)
    m = layer_metrics("polygon", two_countries.features, out, s.graph, shapely.is_valid(orig))
    assert 0 < m["retention"] < 1
    assert m["displacement_max"] <= 20.0 + 1e-9  # DP bounds displacement by the tolerance
    assert m["vertices"] == vertex_count("polygon", out)


def test_lines_and_geojson_reading(tmp_path):
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "a"},
         "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1], [1, 1], [2, 0]]}}]}
    path = tmp_path / "l.geojson"
    path.write_text(json.dumps(fc))
    layer = Layer.from_geojson(path)
    assert layer.kind == "line" and len(layer.features[0][0]) == 3  # duplicate removed
    out = Simplifier(layer, "dp", "independent").simplify(1e9)
    assert len(out[0][0]) == 2


def test_mercator_and_resolution():
    xy = to_mercator(np.array([[0.0, 0.0], [180.0, 0.0]]))
    np.testing.assert_allclose(xy[1, 0], 20037508.342789244)
    assert resolution(0) == pytest.approx(156543.03392804097)
    assert resolution(1) == pytest.approx(resolution(0) / 2)


def test_mvt_encoding():
    assert _zigzag(np.array([0, -1, 1, -2, 2])).tolist() == [0, 1, 2, 3, 4]
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], float) * resolution(0) / 16
    q = quantize(square, 0)
    assert (q[2] - q[0]).tolist() == [10, -10]
    blob = encode_feature("polygon", [[square]], 0)
    # MoveTo + 2 params + LineTo + 6 params + ClosePath, each fits in one byte except large first deltas
    assert blob[0] == 9 and blob[-1] == 15
    assert layer_size("polygon", [[[square]]], 0)["bytes"] == len(blob)
