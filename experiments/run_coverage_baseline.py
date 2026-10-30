"""Compare against GEOS CoverageSimplifier (``shapely.coverage_simplify``).

GEOS/JTS ship a coverage-aware Visvalingam-Whyatt simplifier whose tolerance
is the square root of an area, the same convention as geosimplify's VW. For
each zoom we run it and our shared-arc + repair modes, and score all of them
on the same geometry-only metrics. Displacement here is GEOS's discrete
Hausdorff distance, because the coverage simplifier does not expose which
original vertices it kept.

Writes results/coverage_baseline.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geosimplify import Layer, Simplifier, resolution  # noqa: E402
from geosimplify.encoding import layer_size  # noqa: E402
from geosimplify.metrics import coverage_errors, net_area, to_shapely, vertex_count  # noqa: E402

ZOOMS = (2, 4, 6, 8)


def from_shapely(geoms) -> list:
    """Shapely (Multi)Polygons back to the nested-array feature format."""
    out = []
    for g in geoms:
        polys = [p for p in shapely.get_parts(g) if p.geom_type == "Polygon" and not p.is_empty]
        out.append([[np.asarray(p.exterior.coords)[:, :2]] + [np.asarray(h.coords)[:, :2] for h in p.interiors]
                    for p in polys])
    return out


def polygonal(geoms):
    """``make_valid`` can return geometry collections; keep only their polygons."""
    out = []
    for g in shapely.make_valid(geoms):
        parts = [p for p in shapely.get_parts(g) if p.geom_type == "Polygon"]
        multi = [q for p in shapely.get_parts(g) if p.geom_type == "MultiPolygon" for q in p.geoms]
        out.append(shapely.MultiPolygon(parts + multi))
    return np.array(out, dtype=object)


def score(name, features, orig, orig_valid, orig_union, a0, z, seconds):
    res = resolution(z)
    geoms = to_shapely("polygon", features)
    empty = shapely.is_empty(geoms)
    hd = shapely.hausdorff_distance(orig[~empty], geoms[~empty])
    a1 = np.array([net_area(f) for f in features])
    return {
        "method": name, "zoom": z, "resolution": res, "seconds": seconds,
        "vertices": vertex_count("polygon", features),
        "invalid_features": int((~shapely.is_valid(geoms) & orig_valid & ~empty).sum()),
        "features_dropped": int(empty.sum()),
        "hausdorff_p95_px": float(np.percentile(hd, 95) / res),
        "hausdorff_max_px": float(hd.max() / res),
        "area_change": float(np.abs(a1 - a0).sum() / a0.sum()),
        **coverage_errors(orig_union, geoms),
        **layer_size("polygon", features, z),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()
    layer = Layer.from_geojson(Path(args.data) / "ne_10m_admin_0_countries.geojson")
    orig = to_shapely(layer.kind, layer.features)
    orig_valid = shapely.is_valid(orig)
    orig_union = shapely.union_all(shapely.make_valid(orig))
    a0 = np.array([net_area(f) for f in layer.features])
    n0 = vertex_count("polygon", layer.features)

    ours = {name: Simplifier(layer, method, "safe") for name, method in (("dp_safe", "dp"), ("vw_safe", "vw"))}
    rows = []
    for z in ZOOMS:
        tol = resolution(z)
        t0 = time.time()
        try:
            simplified = shapely.coverage_simplify(polygonal(orig), tol)
            rows.append(score("geos_coverage", from_shapely(simplified), orig, orig_valid, orig_union, a0, z,
                              time.time() - t0))
        except shapely.errors.GEOSException as exc:
            print(f"coverage_simplify failed at z{z}: {exc}", flush=True)
        for name, simp in ours.items():
            t0 = time.time()
            features = simp.simplify(tol)
            rows.append(score(name, features, orig, orig_valid, orig_union, a0, z, time.time() - t0))
        for r in rows[-3:]:
            r["retention"] = r["vertices"] / n0
            print(f"z{z} {r['method']:14s} ret={r['retention']:.3f} hd95={r['hausdorff_p95_px']:.2f}px "
                  f"inval={r['invalid_features']} gaps={r['gap_count']} {r['seconds']:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(Path(args.out) / "coverage_baseline.csv", index=False)


if __name__ == "__main__":
    main()
