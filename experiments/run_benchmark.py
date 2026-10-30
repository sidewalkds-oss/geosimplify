"""Benchmark simplification algorithms and topology modes on Natural Earth data.

Experiments
-----------
zoom    Countries (10m) and coastline (10m) simplified for zooms 2-8 at a
        1-pixel tolerance, for every method x mode: retention, displacement,
        area change, validity, crossings, coverage gaps/overlaps, MVT bytes,
        runtime. -> results/zoom.csv
pareto  Countries (10m, topology mode) over a sweep of tolerances: the
        retention vs displacement trade-off of each method. -> results/pareto.csv

Usage:
    python experiments/run_benchmark.py [--data data] [--only zoom|pareto]
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
from geosimplify.algorithms import METHODS  # noqa: E402
from geosimplify.core import crossing_segments  # noqa: E402
from geosimplify.encoding import layer_size  # noqa: E402
from geosimplify.metrics import coverage_errors, is_valid, layer_metrics, to_shapely  # noqa: E402

ZOOMS = (2, 4, 6, 8)
DATASETS = {"countries": "ne_10m_admin_0_countries.geojson", "coastline": "ne_10m_coastline.geojson"}


def zoom_experiment(data: Path, out: Path):
    path = out / "zoom.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["dataset"], r["method"], r["mode"], r["zoom"]) for r in rows}
    for name, file in DATASETS.items():
        layer = Layer.from_geojson(data / file)
        orig = to_shapely(layer.kind, layer.features)
        orig_valid = is_valid(layer.kind, orig)
        orig_union = shapely.union_all(shapely.make_valid(orig)) if layer.kind == "polygon" else None
        modes = ("independent", "topology", "safe") if layer.kind == "polygon" else ("independent", "safe")
        for method in METHODS:
            for mode in modes:
                if all((name, method, mode, z) in done for z in ZOOMS):
                    continue
                t0 = time.time()
                simp = Simplifier(layer, method, mode)
                build = time.time() - t0
                for z in ZOOMS:
                    if (name, method, mode, z) in done:
                        continue
                    res = resolution(z)
                    t0 = time.time()
                    features = simp.for_zoom(z)
                    elapsed = time.time() - t0
                    row = {"dataset": name, "method": method, "mode": mode, "zoom": z, "resolution": res,
                           "build_seconds": build, "simplify_seconds": elapsed, "repairs": simp.repairs,
                           **layer_metrics(layer.kind, layer.features, features, simp.graph, orig_valid),
                           "crossings": len(crossing_segments(simp.graph)),
                           "side_conflicts": len(simp._side_conflicts()) if layer.kind == "polygon" else 0,
                           **layer_size(layer.kind, features, z)}
                    if orig_union is not None:
                        row.update(coverage_errors(orig_union, to_shapely(layer.kind, features)))
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(path, index=False)
                    print(f"{name:9s} {method:6s} {mode:11s} z{z}  ret={row['retention']:.3f} "
                          f"cross={row['crossings']} {elapsed:5.1f}s", flush=True)
                del simp


def pareto_experiment(data: Path, out: Path):
    layer = Layer.from_geojson(data / DATASETS["countries"])
    orig_valid = is_valid(layer.kind, to_shapely(layer.kind, layer.features))
    tolerances = np.geomspace(100, 100_000, 13)  # metres
    rows = []
    for method in METHODS:
        simp = Simplifier(layer, method, "topology")
        for tol in tolerances:
            features = simp.simplify(tol)
            rows.append({"method": method, "tolerance": tol,
                         **layer_metrics(layer.kind, layer.features, features, simp.graph, orig_valid)})
            print(f"pareto {method:6s} tol={tol:9.0f}  ret={rows[-1]['retention']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(out / "pareto.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument("--only", choices=("zoom", "pareto"))
    args = parser.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.only in (None, "pareto"):
        pareto_experiment(data, out)
    if args.only in (None, "zoom"):
        zoom_experiment(data, out)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
