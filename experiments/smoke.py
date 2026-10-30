"""Quick end-to-end check on Natural Earth 1:50m (used by CI)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geosimplify import Layer, Simplifier, layer_size  # noqa: E402

layer = Layer.from_geojson(ROOT / "data" / "ne_50m_admin_0_countries.geojson")
for mode in ("independent", "topology", "safe"):
    simp = Simplifier(layer, "dp", mode)
    features = simp.for_zoom(4)
    n = sum(len(r) - 1 for f in features for p in f for r in p)
    print(f"{mode:11s} {n:6d} vertices  {layer_size(layer.kind, features, 4)['bytes']:7d} bytes  repairs={simp.repairs}")
