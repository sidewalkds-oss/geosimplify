"""Figures and LaTeX tables for the paper from results/*.csv.

    python experiments/make_figures.py [--results results] [--paper paper] [--data data]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geosimplify import Layer, Simplifier, resolution, to_mercator  # noqa: E402
from geosimplify.metrics import _holes, to_shapely  # noqa: E402

# Fixed categorical order: method colour never depends on rank.
COLOR = {"radial": "#2a78d6", "dp": "#eb6834", "vw": "#1baf7a"}
MODE_COLOR = {"independent": "#2a78d6", "topology": "#eb6834", "safe": "#1baf7a"}
LABEL = {"radial": "Radial distance", "dp": "Douglas–Peucker", "vw": "Visvalingam–Whyatt"}
MODE_LABEL = {"independent": "Independent", "topology": "Shared arcs", "safe": "Shared arcs + repair"}
INK, INK2, GRID, ERROR = "#0b0b0b", "#52514e", "#e4e3df", "#d03b3b"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, out: Path, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


def fig_pareto(pareto: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for method in ("radial", "dp", "vw"):
        g = pareto[pareto.method == method].sort_values("retention")
        axes[0].plot(g.retention * 100, g.displacement_p95 / 1000, color=COLOR[method], lw=2, marker="o", ms=4,
                     label=LABEL[method])
        axes[1].plot(g.retention * 100, g.area_change * 100, color=COLOR[method], lw=2, marker="o", ms=4)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Vertices retained (%)")
    axes[0].set_ylabel("95th-pct. displacement (km)")
    axes[1].set_ylabel("Area change (%)")
    axes[0].set_title("Shape error", fontsize=9, color=INK, loc="left")
    axes[1].set_title("Area error", fontsize=9, color=INK, loc="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    save(fig, out, "fig2_pareto")


def fig_coverage(zoom: pd.DataFrame, out: Path):
    sub = zoom[zoom.dataset == "countries"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True)
    metrics = [("gap_count", "Sliver gaps (count)"), ("overlap_px", "Overlap area (px²)"), ("crossings", "Crossing segments")]
    sub = sub.assign(overlap_px=sub.overlap_area / sub.resolution**2)
    for ax, (col, title) in zip(axes, metrics):
        for mode in ("independent", "topology", "safe"):
            g = sub[(sub["mode"] == mode) & (sub.method == "dp")].sort_values("zoom")
            ax.plot(g.zoom, g[col] + 1, color=MODE_COLOR[mode], lw=2, marker="o", ms=4, label=MODE_LABEL[mode])
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Zoom level")
        ax.set_xticks(sorted(sub.zoom.unique()))
    axes[0].set_ylabel("Value + 1 (log scale)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    save(fig, out, "fig3_coverage_errors")


def fig_bytes(zoom: pd.DataFrame, data: Path, out: Path):
    from geosimplify.encoding import layer_size

    layer = Layer.from_geojson(data / "ne_10m_admin_0_countries.geojson")
    raw = {z: layer_size(layer.kind, layer.features, z)["bytes"] for z in sorted(zoom.zoom.unique())}
    sub = zoom[(zoom.dataset == "countries") & (zoom["mode"] == "safe")]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for method in ("radial", "dp", "vw"):
        g = sub[sub.method == method].sort_values("zoom")
        ax.plot(g.zoom, g.bytes / g.zoom.map(raw) * 100, color=COLOR[method], lw=2, marker="o", ms=4,
                label=LABEL[method])
    ax.set_xlabel("Zoom level")
    ax.set_ylabel("Geometry bytes (% of unsimplified)")
    ax.set_xticks(sorted(sub.zoom.unique()))
    ax.legend(fontsize=8)
    save(fig, out, "fig4_bytes")
    return raw


def fig_example(data: Path, out: Path):
    """Independent vs shared-arc DP simplification around the Balkans at zoom 4."""
    layer = Layer.from_geojson(data / "ne_10m_admin_0_countries.geojson")
    lo = to_mercator(np.array([[12.5, 39.5], [30.5, 48.5]]))
    box = shapely.box(lo[0, 0], lo[0, 1], lo[1, 0], lo[1, 1])
    orig = to_shapely(layer.kind, layer.features)
    keep = np.flatnonzero(shapely.intersects(orig, box))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, mode in zip(axes, ("independent", "topology", "safe")):
        geoms = to_shapely(layer.kind, Simplifier(layer, "dp", mode).for_zoom(4))
        geoms = shapely.make_valid(geoms[keep])
        # Gaps as the metric defines them: holes in the simplified union that
        # touch no hole of the original union (land lost at the coast is not a gap).
        orig_holes = shapely.union_all(_holes(shapely.union_all(shapely.make_valid(orig[keep]))))
        new_holes = [h for h in _holes(shapely.union_all(geoms)) if not h.intersects(orig_holes)]
        gaps = shapely.intersection(shapely.union_all(new_holes), box) if new_holes else None
        overlaps = []
        for i in range(len(geoms)):
            for j in range(i + 1, len(geoms)):
                inter = shapely.intersection(geoms[i], geoms[j])
                if shapely.area(inter) > 0:
                    overlaps.append(inter)
        for g in shapely.get_parts(shapely.intersection(geoms, box)):
            for poly in shapely.get_parts(g):
                if poly.geom_type == "Polygon":
                    ax.fill(*poly.exterior.xy, fc="#f0efec", ec=INK2, lw=0.4)
        for geom, colour in ((gaps, ERROR), (shapely.union_all(overlaps) if overlaps else None, "#2a78d6")):
            if geom is None:
                continue
            for poly in shapely.get_parts(geom):
                if poly.geom_type == "Polygon" and poly.area > 0:
                    ax.fill(*poly.exterior.xy, fc=colour, ec=colour, lw=0.6)
        ax.set_xlim(lo[0, 0], lo[1, 0])
        ax.set_ylim(lo[0, 1], lo[1, 1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(GRID)
        ax.set_title(MODE_LABEL[mode], fontsize=9, color=INK, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ERROR), plt.Rectangle((0, 0), 1, 1, color="#2a78d6")]
    fig.legend(handles, ["gap (land covered by no country)", "overlap (land in two countries)"],
               loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.08))
    save(fig, out, "fig1_example")


def table_zoom(zoom: pd.DataFrame, dataset: str, z: int, raw_bytes: dict | None = None) -> str:
    sub = zoom[(zoom.dataset == dataset) & (zoom.zoom == z)]
    modes = ["independent", "topology", "safe"] if dataset == "countries" else ["independent", "safe"]
    cols = r"Method & Mode & Retained & Disp.\ p95 (px) & Invalid & Crossings"
    if dataset == "countries":
        cols += r" & Gaps & Overlap (px$^2$)"
    cols += r" & KB & Time (s) \\"
    rows = []
    for method in ("radial", "dp", "vw"):
        for mode in modes:
            r = sub[(sub.method == method) & (sub["mode"] == mode)].iloc[0]
            cells = [LABEL[method] if mode == modes[0] else "", MODE_LABEL[mode], f"{r.retention * 100:.1f}\\%",
                     f"{r.displacement_p95 / r.resolution:.2f}", f"{int(r.invalid_features)}", f"{int(r.crossings)}"]
            if dataset == "countries":
                cells += [f"{int(r.gap_count)}", f"{r.overlap_area / r.resolution**2:.1f}"]
            cells += [f"{r.bytes / 1024:.0f}", f"{r.simplify_seconds:.1f}"]
            rows.append(" & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    ncol = 10 if dataset == "countries" else 8
    return "\n".join([rf"\begin{{tabular}}{{ll{'r' * (ncol - 2)}}}", r"\toprule", cols, r"\midrule", *rows,
                      r"\bottomrule", r"\end{tabular}"])


def table_repair_cost(zoom: pd.DataFrame) -> str:
    sub = zoom[zoom.dataset == "countries"]
    rows = []
    for method in ("radial", "dp", "vw"):
        cells = [LABEL[method]]
        for z in sorted(sub.zoom.unique()):
            t = sub[(sub.method == method) & (sub["mode"] == "topology") & (sub.zoom == z)].iloc[0]
            s = sub[(sub.method == method) & (sub["mode"] == "safe") & (sub.zoom == z)].iloc[0]
            cells.append(f"+{(s.vertices / t.vertices - 1) * 100:.1f}\\% / {int(s.repairs)}")
        rows.append(" & ".join(cells) + r" \\")
    zs = sorted(sub.zoom.unique())
    return "\n".join([r"\begin{tabular}{l" + "c" * len(zs) + "}", r"\toprule",
                      "Method & " + " & ".join(f"z{z}" for z in zs) + r" \\", r"\midrule", *rows,
                      r"\bottomrule", r"\end{tabular}"])


def table_equal_budget(pareto: pd.DataFrame, budgets=(0.05, 0.1, 0.25)) -> tuple[str, list]:
    """Displacement (p95, km) of each method interpolated at equal vertex retention."""
    rows, raw = [], []
    for method in ("radial", "dp", "vw"):
        g = pareto[pareto.method == method].sort_values("retention")
        cells = [LABEL[method]]
        for b in budgets:
            v = np.exp(np.interp(np.log(b), np.log(g.retention), np.log(g.displacement_p95))) / 1000
            a = np.exp(np.interp(np.log(b), np.log(g.retention), np.log(g.area_change))) * 100
            cells.append(f"{v:.1f} km / {a:.2f}\\%")
            raw.append((method, b, v, a))
        rows.append(" & ".join(cells) + r" \\")
    return "\n".join([r"\begin{tabular}{l" + "c" * len(budgets) + "}", r"\toprule",
                      "Method & " + " & ".join(f"{b * 100:g}\\% of vertices" for b in budgets) + r" \\",
                      r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]), raw


def table_baseline(base: pd.DataFrame) -> str:
    names = {"geos_coverage": "GEOS CoverageSimplifier", "dp_safe": "Douglas--Peucker + repair",
             "vw_safe": "Visvalingam--Whyatt + repair"}
    rows = []
    for z in sorted(base.zoom.unique()):
        for key, name in names.items():
            r = base[(base.method == key) & (base.zoom == z)].iloc[0]
            rows.append(" & ".join([f"z{z}" if key == "geos_coverage" else "", name, f"{r.vertices / 1000:.1f}k",
                                    f"{r.hausdorff_p95_px:.2f}", f"{r.hausdorff_max_px:.1f}",
                                    f"{int(r.invalid_features)}", f"{int(r.features_dropped)}",
                                    f"{int(r.gap_count)}", f"{r.bytes / 1024:.0f}", f"{r.seconds:.1f}"]) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    return "\n".join([r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
                      r"Zoom & Method & Vertices & Hausd.\ p95 (px) & Hausd.\ max (px) & Invalid & Dropped & Gaps & KB & Time (s) \\",
                      r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--paper", default=str(ROOT / "paper"))
    parser.add_argument("--data", default=str(ROOT / "data"))
    args = parser.parse_args()
    res, data = Path(args.results), Path(args.data)
    figs, tabs = Path(args.paper) / "figures", Path(args.paper) / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    zoom = pd.read_csv(res / "zoom.csv")
    pareto = pd.read_csv(res / "pareto.csv")
    fig_pareto(pareto, figs)
    fig_coverage(zoom, figs)
    raw = fig_bytes(zoom, data, figs)
    fig_example(data, figs)
    (tabs / "zoom_countries.tex").write_text(table_zoom(zoom, "countries", 4), encoding="utf-8")
    (tabs / "zoom_coastline.tex").write_text(table_zoom(zoom, "coastline", 4), encoding="utf-8")
    (tabs / "repair_cost.tex").write_text(table_repair_cost(zoom), encoding="utf-8")
    (tabs / "baseline.tex").write_text(table_baseline(pd.read_csv(res / "coverage_baseline.csv")), encoding="utf-8")
    eq_tex, eq_raw = table_equal_budget(pareto)
    (tabs / "equal_budget.tex").write_text(eq_tex, encoding="utf-8")
    print("unsimplified bytes by zoom:", raw)
    for row in eq_raw:
        print("equal budget", row)


if __name__ == "__main__":
    main()
