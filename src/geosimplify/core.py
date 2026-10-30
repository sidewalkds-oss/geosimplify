"""High-level API: simplify a layer for display at a zoom level."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely

from .algorithms import _segment_distance
from .metrics import is_valid, to_shapely
from .topology import ArcGraph, read_geojson, resolution

MODES = ("independent", "topology", "safe")


def _point_segments_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from point ``p`` to each segment ``a[k]-b[k]``."""
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    t = np.where(denom > 0, np.einsum("ij,ij->i", p - a, ab) / np.where(denom > 0, denom, 1), 0.0)
    proj = a + np.clip(t, 0, 1)[:, None] * ab
    return np.hypot(*(p - proj).T)


def crossing_segments(graph: ArcGraph) -> list[tuple[int, int, int]]:
    """Simplified segments that cross or overlap another simplified segment.

    Returns ``(arc, i, j)`` triples: the segment joins original vertices ``i``
    and ``j`` of ``arc``. Segments that merely touch at shared endpoints
    (consecutive segments, arcs meeting at a junction) are not counted.
    """
    starts, ends, meta = [], [], []
    for a, (arc, keep) in enumerate(zip(graph.arcs, graph.keep_masks)):
        idx = np.flatnonzero(keep)
        if len(idx) >= 2:
            starts.append(arc.coords[idx[:-1]])
            ends.append(arc.coords[idx[1:]])
            meta.append(np.column_stack([np.full(len(idx) - 1, a), idx[:-1], idx[1:]]))
    segs = shapely.linestrings(np.stack([np.vstack(starts), np.vstack(ends)], axis=1))
    meta = np.vstack(meta)
    tree = shapely.STRtree(segs)
    pairs = np.hstack([tree.query(segs, predicate=p) for p in ("crosses", "overlaps")])
    hit = np.unique(pairs[:, pairs[0] != pairs[1]].ravel())
    return [tuple(map(int, m)) for m in meta[hit]]


@dataclass
class Layer:
    kind: str  # "polygon" or "line"
    features: list
    properties: list[dict]

    @classmethod
    def from_geojson(cls, path) -> "Layer":
        return cls(*read_geojson(path))


class Simplifier:
    """Simplify a layer with a given algorithm and topology mode.

    Parameters
    ----------
    method:
        ``"dp"`` (Douglas-Peucker), ``"vw"`` (Visvalingam-Whyatt) or ``"radial"``.
    mode:
        ``"independent"``: each ring or line separately.
        ``"topology"``: shared borders are simplified once (no gaps or
        overlaps from diverging borders).
        ``"safe"``: topology mode plus repair. Each round finds (a) simplified
        segments that cross another simplified segment, in the same feature or
        a neighbour, and (b) for polygons, rings that changed side: an
        interior point of ring R that is now inside (or no longer inside)
        another ring S, as when a simplified coastline swallows an island. For
        each conflict it restores the original vertex that deviates most from
        the offending segment (for (b), the segment of S nearest the point).
        Rounds repeat until no conflicts remain or ``max_repairs`` have run.
    """

    def __init__(self, layer: Layer, method: str = "dp", mode: str = "topology", max_repairs: int = 30):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.layer = layer
        self.method = method
        self.mode = mode
        self.max_repairs = max_repairs
        self.graph = ArcGraph.build(layer.kind, layer.features, topology=mode != "independent")
        self._orig_valid = None
        self._sides = None
        self.repairs = 0  # vertices restored by the last call
        self.remaining_crossings = 0
        self.remaining_side_conflicts = 0

    @property
    def original_valid(self) -> np.ndarray:
        if self._orig_valid is None:
            self._orig_valid = is_valid(self.layer.kind, to_shapely(self.layer.kind, self.layer.features))
        return self._orig_valid

    def simplify(self, tol: float) -> list:
        """Simplified features at distance tolerance ``tol`` (metres)."""
        self.repairs = 0
        out = self.graph.simplify(self.method, tol)
        if self.mode != "safe":
            return out
        extra = [np.zeros(len(a.coords), dtype=bool) for a in self.graph.arcs]
        for round_ in range(self.max_repairs + 1):
            crossings = crossing_segments(self.graph)
            sides = self._side_conflicts() if self.layer.kind == "polygon" else []
            self.remaining_crossings, self.remaining_side_conflicts = len(crossings), len(sides)
            if round_ == self.max_repairs:
                break
            restored = 0
            for a, i, j in crossings + sides:
                if j - i < 2:
                    continue  # an original segment: nothing to restore
                coords = self.graph.arcs[a].coords
                d = _segment_distance(coords[i + 1 : j], coords[i], coords[j])
                k = i + 1 + int(d.argmax())
                if not extra[a][k]:
                    extra[a][k] = True
                    restored += 1
            if restored == 0:
                break
            self.repairs += restored
            out = self.graph.simplify(self.method, tol, extra)
        return out

    def _ring_polygons(self, path_coords: list[np.ndarray]) -> np.ndarray:
        polys = [shapely.Polygon(c) if len(c) >= 4 and len(np.unique(c[:-1], axis=0)) >= 3 else shapely.Polygon()
                 for c in path_coords]
        return np.array(polys, dtype=object)

    def _side_conflicts(self) -> list[tuple[int, int, int]]:
        """Segments to refine where a ring's interior point changed side of another ring."""
        g = self.graph
        if self._sides is None:
            rings = self._ring_polygons(g.path_coords([a.coords for a in g.arcs]))
            ok = ~shapely.is_empty(rings)
            points = np.array([shapely.point_on_surface(r) if o else None for r, o in zip(rings, ok)], dtype=object)
            idx = np.flatnonzero(ok)
            pairs = shapely.STRtree(rings[idx]).query(points[idx], predicate="within")
            orig = {(int(idx[p]), int(idx[s])) for p, s in pairs.T if idx[p] != idx[s]}
            self._sides = (points, idx, orig)
        points, idx, orig = self._sides
        coords = g.path_coords([a.coords[k] for a, k in zip(g.arcs, g.keep_masks)])
        rings = self._ring_polygons(coords)
        live = idx[~shapely.is_empty(rings[idx])]
        pairs = shapely.STRtree(rings[live]).query(points[live], predicate="within")
        now = {(int(live[p]), int(live[s])) for p, s in pairs.T if live[p] != live[s]}
        out = []
        for p, s in (orig ^ now):
            # Refine ring s at its simplified segment nearest the point of ring p.
            pt = np.array(points[p].coords[0])
            best = None
            for a, _ in g.paths[s][1]:
                kept = np.flatnonzero(g.keep_masks[a])
                if len(kept) < 2:
                    continue
                c = g.arcs[a].coords
                d = _point_segments_distance(pt, c[kept[:-1]], c[kept[1:]])
                k = int(d.argmin())
                if best is None or d[k] < best[0]:
                    best = (float(d[k]), a, int(kept[k]), int(kept[k + 1]))
            if best is not None:
                out.append(best[1:])
        return out

    def for_zoom(self, zoom: float, pixels: float = 1.0) -> list:
        """Simplify so that the tolerance is ``pixels`` screen pixels at ``zoom``."""
        return self.simplify(pixels * resolution(zoom))
