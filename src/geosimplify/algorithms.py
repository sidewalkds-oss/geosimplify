"""Line simplification algorithms.

Douglas-Peucker and Visvalingam-Whyatt are computed as *significance* arrays:
one number per vertex such that simplifying at tolerance ``t`` keeps exactly
the vertices with significance ``> t`` (DP) or ``>= t`` (VW, an area). Computing
the significance once and thresholding it is equivalent to re-running the
algorithm at every tolerance, and makes tolerance sweeps and per-zoom
simplification cheap. Endpoints always have infinite significance.

Radial distance is not hierarchical and is run directly.

All functions take an ``(n, 2)`` float array of projected coordinates and treat
it as an open polyline whose two endpoints are fixed.
"""

from __future__ import annotations

import heapq

import numpy as np


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point to the closed segment a-b."""
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return np.hypot(*(points - a).T)
    t = np.clip((points - a) @ ab / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.hypot(*(points - proj).T)


def dp_significance(coords: np.ndarray) -> np.ndarray:
    """Douglas-Peucker significance of every vertex.

    A vertex's significance is the deviation at which DP inserts it, capped by
    its ancestors' significance (so the result is monotone along the
    recursion). Keeping ``sig > t`` reproduces DP at tolerance ``t`` exactly.
    """
    n = len(coords)
    sig = np.zeros(n)
    sig[0] = sig[-1] = np.inf
    stack = [(0, n - 1, np.inf)]
    while stack:
        i, j, cap = stack.pop()
        if j <= i + 1:
            continue
        d = _segment_distance(coords[i + 1 : j], coords[i], coords[j])
        k = int(d.argmax())
        idx = i + 1 + k
        s = min(float(d[k]), cap)
        sig[idx] = s
        stack.append((i, idx, s))
        stack.append((idx, j, s))
    return sig


def _triangle_area(a, b, c) -> float:
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def vw_significance(coords: np.ndarray) -> np.ndarray:
    """Visvalingam-Whyatt effective area of every vertex.

    Vertices are removed smallest-area first; when a removal changes a
    neighbour's triangle, the neighbour's area is raised to at least the
    removed area, which keeps the elimination order consistent with
    thresholding. Keeping ``sig >= t`` reproduces VW with area tolerance ``t``.
    """
    n = len(coords)
    sig = np.full(n, np.inf)
    if n <= 2:
        return sig
    pts = coords.tolist()
    prev = list(range(-1, n - 1))
    nxt = list(range(1, n + 1))
    area = [np.inf] * n
    heap = []
    for i in range(1, n - 1):
        area[i] = _triangle_area(pts[i - 1], pts[i], pts[i + 1])
        heap.append((area[i], i))
    heapq.heapify(heap)
    removed = [False] * n
    floor = 0.0
    while heap:
        a, i = heapq.heappop(heap)
        if removed[i] or a != area[i]:
            continue  # stale entry
        floor = max(floor, a)
        sig[i] = floor
        removed[i] = True
        p, q = prev[i], nxt[i]
        nxt[p], prev[q] = q, p
        for j in (p, q):
            if 0 < j < n - 1:
                area[j] = _triangle_area(pts[prev[j]], pts[j], pts[nxt[j]])
                heapq.heappush(heap, (area[j], j))
    return sig


def radial_distance(coords: np.ndarray, tol: float) -> np.ndarray:
    """Keep a vertex only if it is at least ``tol`` from the last kept vertex."""
    n = len(coords)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    last = coords[0]
    tol2 = tol * tol
    for i in range(1, n - 1):
        dx, dy = coords[i, 0] - last[0], coords[i, 1] - last[1]
        if dx * dx + dy * dy >= tol2:
            keep[i] = True
            last = coords[i]
    return keep


SIGNIFICANCE = {"dp": dp_significance, "vw": vw_significance}
METHODS = ("radial", "dp", "vw")


def keep_mask(coords: np.ndarray, method: str, tol: float, sig: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of retained vertices.

    ``tol`` is a distance for every method; for VW it is converted to the area
    threshold ``tol**2`` (the area of a square one tolerance wide), a common
    convention that puts the three methods on a similar scale. Pass a
    precomputed ``sig`` to avoid recomputing DP/VW significance.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if method == "radial":
        return radial_distance(coords, tol)
    if sig is None:
        sig = SIGNIFICANCE[method](coords)
    if method == "dp":
        return sig > tol
    return sig >= tol * tol


def simplify_line(coords: np.ndarray, method: str, tol: float) -> np.ndarray:
    """Simplify an open polyline; endpoints are kept."""
    return coords[keep_mask(coords, method, tol)]
