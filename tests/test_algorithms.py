import numpy as np
import pytest

from geosimplify.algorithms import dp_significance, keep_mask, radial_distance, simplify_line, vw_significance


def naive_dp(coords, tol):
    """Textbook recursive Douglas-Peucker, used as a reference."""
    def rec(i, j, keep):
        if j <= i + 1:
            return
        a, b = coords[i], coords[j]
        ab = b - a
        t = np.clip(((coords[i + 1 : j] - a) @ ab) / (ab @ ab), 0, 1) if ab @ ab else np.zeros(j - i - 1)
        d = np.hypot(*(coords[i + 1 : j] - (a + t[:, None] * ab)).T)
        k = int(d.argmax())
        if d[k] > tol:
            keep[i + 1 + k] = True
            rec(i, i + 1 + k, keep)
            rec(i + 1 + k, j, keep)
    keep = np.zeros(len(coords), bool)
    keep[0] = keep[-1] = True
    rec(0, len(coords) - 1, keep)
    return keep


def _cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


def naive_vw(coords, area_tol):
    """Textbook Visvalingam-Whyatt: repeatedly drop the smallest triangle below the threshold."""
    idx = list(range(len(coords)))
    while len(idx) > 2:
        areas = [0.5 * abs(_cross(coords[idx[k]] - coords[idx[k - 1]], coords[idx[k + 1]] - coords[idx[k - 1]]))
                 for k in range(1, len(idx) - 1)]
        k = int(np.argmin(areas))
        if areas[k] >= area_tol:
            break
        del idx[k + 1]
    keep = np.zeros(len(coords), bool)
    keep[idx] = True
    return keep


@pytest.fixture
def wiggle():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 100, 300)
    return np.column_stack([x, 5 * np.sin(x / 3) + rng.normal(0, 1, len(x))])


@pytest.mark.parametrize("tol", [0.1, 0.5, 1.0, 2.0, 5.0, 20.0])
def test_dp_significance_matches_recursive_dp(wiggle, tol):
    np.testing.assert_array_equal(keep_mask(wiggle, "dp", tol), naive_dp(wiggle, tol))


@pytest.mark.parametrize("tol", [0.3, 1.0, 2.0, 4.0])
def test_vw_significance_matches_iterative_vw(wiggle, tol):
    np.testing.assert_array_equal(keep_mask(wiggle, "vw", tol), naive_vw(wiggle, tol * tol))


def test_endpoints_always_kept(wiggle):
    for method in ("dp", "vw", "radial"):
        keep = keep_mask(wiggle, method, 1e9)
        assert keep[0] and keep[-1] and keep.sum() == 2


def test_dp_bounds_deviation(wiggle):
    tol = 2.0
    out = simplify_line(wiggle, "dp", tol)
    # every original vertex is within tol of the simplified polyline
    from shapely.geometry import LineString, Point
    line = LineString(out)
    assert max(line.distance(Point(p)) for p in wiggle) <= tol + 1e-9


def test_radial_spacing(wiggle):
    keep = radial_distance(wiggle, 3.0)
    kept = wiggle[keep][:-1]
    assert (np.hypot(*np.diff(kept, axis=0).T) >= 3.0).all()


def test_significance_monotone_in_tolerance(wiggle):
    for method in ("dp", "vw"):
        counts = [keep_mask(wiggle, method, t).sum() for t in (0.1, 0.5, 1, 2, 5)]
        assert counts == sorted(counts, reverse=True)


def test_degenerate_inputs():
    two = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert dp_significance(two).tolist() == [np.inf, np.inf]
    assert vw_significance(two).tolist() == [np.inf, np.inf]
    same = np.zeros((4, 2))
    assert keep_mask(same, "dp", 1.0).tolist() == [True, False, False, True]


def test_unknown_method(wiggle):
    with pytest.raises(ValueError):
        keep_mask(wiggle, "bogus", 1.0)
