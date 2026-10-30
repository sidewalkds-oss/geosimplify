"""Quality metrics for simplified layers (Shapely is used for validity and distances)."""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import MultiLineString, MultiPolygon, Polygon


def to_shapely(kind: str, features: list) -> np.ndarray:
    """One Shapely geometry per feature (empty if everything collapsed)."""
    out = []
    for feat in features:
        if kind == "polygon":
            polys = [Polygon(p[0], p[1:]) for p in feat if p]
            out.append(MultiPolygon(polys) if polys else MultiPolygon())
        else:
            out.append(MultiLineString([l for l in feat if len(l) >= 2]))
    return np.array(out, dtype=object)


def vertex_count(kind: str, features: list) -> int:
    """Stored vertices (closing vertices of rings excluded)."""
    if kind == "polygon":
        return sum(len(r) - 1 for feat in features for p in feat for r in p)
    return sum(len(l) for feat in features for l in feat)


def _ring_area(r: np.ndarray) -> float:
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def net_area(feat: list) -> float:
    """Exterior minus hole area by the shoelace formula; valid even for invalid rings."""
    return sum(abs(_ring_area(p[0])) - sum(abs(_ring_area(h)) for h in p[1:]) for p in feat if p)


def is_valid(kind: str, geoms: np.ndarray) -> np.ndarray:
    return shapely.is_valid(geoms) if kind == "polygon" else shapely.is_simple(geoms)


def layer_metrics(kind: str, original: list, simplified: list, graph, orig_valid: np.ndarray) -> dict:
    """Retention, displacement, area change and validity of one simplified layer.

    ``graph`` is the :class:`~geosimplify.topology.ArcGraph` that produced
    ``simplified``. Displacements are in metres; divide by the zoom resolution
    for screen pixels. Features that collapsed entirely are counted in
    ``features_dropped`` and excluded from the displacement statistics.
    """
    geoms = to_shapely(kind, simplified)
    empty = shapely.is_empty(geoms)
    arc_disp = graph.displacement()
    disp = np.array([arc_disp[a].max() if len(a) else 0.0 for a in graph.feature_arcs])
    disp = disp[~empty]
    valid = is_valid(kind, geoms)
    out = {
        "vertices": vertex_count(kind, simplified),
        "vertices_original": vertex_count(kind, original),
        "features_dropped": int(empty.sum()),
        "displacement_median": float(np.median(disp)),
        "displacement_p95": float(np.percentile(disp, 95)),
        "displacement_max": float(disp.max()),
        "invalid_features": int((~valid & orig_valid & ~empty).sum()),
    }
    if kind == "polygon":
        a0 = np.array([net_area(f) for f in original])
        a1 = np.array([net_area(f) for f in simplified])
        out["area_change"] = float(np.abs(a1 - a0).sum() / a0.sum())
    out["retention"] = out["vertices"] / out["vertices_original"]
    return out


def _holes(geom) -> list:
    polys = [p for p in shapely.get_parts(geom) if p.geom_type == "Polygon" and not p.is_empty]
    return [Polygon(h) for p in polys for h in p.interiors]


def coverage_errors(orig_union, geoms: np.ndarray) -> dict:
    """Overlaps and sliver gaps between neighbouring polygons.

    *Overlap* is the area covered by more than one feature (sum of areas minus
    area of the union). A *gap* is a hole in the union of simplified features
    that touches no hole of the original union: a sliver that opened along a
    shared border. Geometries are repaired with ``make_valid`` first.
    """
    fixed = shapely.make_valid(geoms[~shapely.is_empty(geoms)])
    fixed = shapely.get_parts(fixed)
    fixed = fixed[shapely.get_type_id(fixed) == 3]  # polygons only
    union = shapely.union_all(fixed)
    overlap = float(shapely.area(fixed).sum() - union.area)
    old = shapely.union_all(_holes(orig_union)) if _holes(orig_union) else Polygon()
    new = [h for h in _holes(union) if not h.intersects(old)]
    return {
        "overlap_area": max(overlap, 0.0),
        "gap_count": len(new),
        "gap_area": float(sum(h.area for h in new)),
    }
