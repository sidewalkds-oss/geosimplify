"""Download the Natural Earth layers used by the benchmark into data/.

Natural Earth is public domain (https://www.naturalearthdata.com). Files come
from the project's GitHub mirror of its GeoJSON exports.

    python experiments/download_data.py [--small]
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
FILES = ["ne_10m_admin_0_countries.geojson", "ne_10m_coastline.geojson", "ne_50m_admin_0_countries.geojson"]
SMALL = ["ne_50m_admin_0_countries.geojson", "ne_50m_coastline.geojson"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="only the 1:50m layers (for quick checks)")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in SMALL if args.small else FILES:
        target = out / name
        if target.exists():
            print(f"have {name}")
            continue
        print(f"downloading {name}")
        urllib.request.urlretrieve(BASE + name, target)


if __name__ == "__main__":
    main()
