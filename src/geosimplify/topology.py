"""Geometry model, Web Mercator projection, and arc topology.

A *layer* is a list of features; a polygon feature is a list of polygons, each a
list of closed rings; a line feature is a list of open or closed lines. All
coordinates are ``(n, 2)`` float arrays in Web Mercator metres.

Simplification works on *arcs*: polylines whose two endpoints are fixed.

* ``independent`` mode: every ring or line is its own arc. Neighbouring
  polygons are simplified separately, so their shared border diverges.
* ``topology`` mode: rings are cut at *junctions* (vertices where more than two
  boundary paths meet), identical arcs are stored once, and every ring that
  uses an arc gets the same simplified version, as in TopoJSON.

A closed arc (an island ring, or a ring with no junctions) also pins two more
vertices that span a triangle, so it can never collapse to a line. Rings built
from several arcs can still collapse; those are dropped and counted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .algorithms import SIGNIFICANCE, keep_mask

EARTH_RADIUS = 6378137.0
MAX_LAT = 85.0511287798066


def to_mercator(lonlat: np.ndarray) -> np.ndarray:
    lon = np.radians(lonlat[:, 0])
    lat = np.radians(np.clip(lonlat[:, 1], -MAX_LAT, MAX_LAT))
    return np.column_stack([EARTH_RADIUS * lon, EARTH_RADIUS * np.log(np.tan(np.pi / 4 + lat / 2))])


def resolution(zoom: float, tile_px: int = 256) -> float:
    """Metres per screen pixel at the equator at ``zoom``."""
    return 2 * np.pi * EARTH_RADIUS / (tile_px * 2**zoom)


# ---------------------------------------------------------------- reading


def _dedupe(coords: np.ndarray) -> np.ndarray:
    keep = np.ones(len(coords), dtype=bool)
    keep[1:] = np.any(coords[1:] != coords[:-1], axis=1)
    return coords[keep]


def read_geojson(path: str | Path) -> tuple[str, list, list[dict]]:
    """Read Polygon/MultiPolygon or LineString/MultiLineString features.

    Returns ``(kind, features, properties)`` with ``kind`` "polygon" or "line".
    Consecutive duplicate vertices are removed.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    features, props, kinds = [], [], set()
    for ft in data["features"]:
        geom = ft["geometry"]
        if geom is None:
            continue
        t, c = geom["type"], geom["coordinates"]
        if t in ("Polygon", "MultiPolygon"):
            kinds.add("polygon")
            polys = [c] if t == "Polygon" else c
            features.append([[_dedupe(to_mercator(np.asarray(r, dtype=float)[:, :2])) for r in p] for p in polys])
        elif t in ("LineString", "MultiLineString"):
            kinds.add("line")
            lines = [c] if t == "LineString" else c
            features.append([_dedupe(to_mercator(np.asarray(l, dtype=float)[:, :2])) for l in lines])
        else:
            continue
        props.append(ft.get("properties") or {})
    if len(kinds) != 1:
        raise ValueError(f"expected one geometry kind, found {kinds}")
    return kinds.pop(), features, props


def paths_of(kind: str, features: list) -> list[tuple[tuple, np.ndarray]]:
    """Flatten a layer into ``(address, coords)`` paths; address locates each path."""
    out = []
    for f, feat in enumerate(features):
        if kind == "polygon":
            for p, poly in enumerate(feat):
                for r, ring in enumerate(poly):
                    out.append(((f, p, r), ring))
        else:
            for l, line in enumerate(feat):
                out.append(((f, l), line))
    return out


# ---------------------------------------------------------------- arcs


def _key(pt) -> tuple:
    return (float(pt[0]), float(pt[1]))


def find_junctions(paths: list[np.ndarray]) -> set[tuple]:
    """Vertices where boundary paths meet, branch or end."""
    neighbours: dict[tuple, set] = {}
    for coords in paths:
        closed = len(coords) > 2 and np.array_equal(coords[0], coords[-1])
        pts = [_key(p) for p in (coords[:-1] if closed else coords)]
        m = len(pts)
        for i, p in enumerate(pts):
            s = neighbours.setdefault(p, set())
            if closed:
                s.add(pts[i - 1])
                s.add(pts[(i + 1) % m])
            else:
                if i > 0:
                    s.add(pts[i - 1])
                if i < m - 1:
                    s.add(pts[i + 1])
    junctions = {p for p, s in neighbours.items() if len(s) > 2}
    for coords in paths:  # ends of open lines are always junctions
        if not np.array_equal(coords[0], coords[-1]):
            junctions.add(_key(coords[0]))
            junctions.add(_key(coords[-1]))
    return junctions


@dataclass
class Arc:
    coords: np.ndarray
    closed: bool
    fixed: np.ndarray = field(default=None)  # vertices that may never be removed
    sig: dict = field(default_factory=dict)  # method -> significance array

    def __post_init__(self):
        if self.fixed is None:
            self.fixed = np.zeros(len(self.coords), dtype=bool)
            self.fixed[0] = self.fixed[-1] = True
            if self.closed and len(self.coords) >= 4:
                # Pin a triangle: the start, the vertex farthest from it, and
                # the vertex farthest from the line joining those two.
                c = self.coords
                far = int(np.hypot(*(c - c[0]).T).argmax())
                self.fixed[far] = True
                u, v = c[far] - c[0], c - c[0]
                d = np.abs(u[0] * v[:, 1] - u[1] * v[:, 0])
                self.fixed[int(d.argmax())] = True

    def significance(self, method: str) -> np.ndarray:
        """Significance computed piecewise between fixed vertices."""
        if method not in self.sig:
            sig = np.full(len(self.coords), np.inf)
            anchors = np.flatnonzero(self.fixed)
            for a, b in zip(anchors[:-1], anchors[1:]):
                part = SIGNIFICANCE[method](self.coords[a : b + 1])
                sig[a + 1 : b] = part[1:-1]
            self.sig[method] = sig
        return self.sig[method]

    def keep(self, method: str, tol: float) -> np.ndarray:
        if method == "radial":
            keep = np.zeros(len(self.coords), dtype=bool)
            anchors = np.flatnonzero(self.fixed)
            for a, b in zip(anchors[:-1], anchors[1:]):
                keep[a : b + 1] |= keep_mask(self.coords[a : b + 1], "radial", tol)
            return keep
        return keep_mask(self.coords, method, tol, self.significance(method)) | self.fixed


@dataclass
class ArcGraph:
    """Paths of a layer expressed as sequences of (arc index, reversed) pairs."""

    kind: str
    arcs: list[Arc]
    paths: list[tuple[tuple, list[tuple[int, bool]]]]  # (address, [(arc, reversed), ...])

    @classmethod
    def build(cls, kind: str, features: list, topology: bool) -> "ArcGraph":
        paths = paths_of(kind, features)
        if not topology:
            arcs = [Arc(c, closed=np.array_equal(c[0], c[-1])) for _, c in paths]
            return cls(kind, arcs, [(addr, [(i, False)]) for i, (addr, _) in enumerate(paths)])

        junctions = find_junctions([c for _, c in paths])
        arcs: list[Arc] = []
        index: dict[tuple, tuple[int, bool]] = {}

        def add(seg: np.ndarray) -> tuple[int, bool]:
            fwd = tuple(map(_key, seg))
            if fwd in index:
                return index[fwd]
            rev = fwd[::-1]
            if rev in index:
                i, r = index[rev]
                return i, not r
            arcs.append(Arc(seg.copy(), closed=fwd[0] == fwd[-1]))
            index[fwd] = (len(arcs) - 1, False)
            return index[fwd]

        out = []
        for addr, coords in paths:
            closed = np.array_equal(coords[0], coords[-1])
            pts = coords[:-1] if closed else coords
            cut = [i for i, p in enumerate(pts) if _key(p) in junctions]
            if closed and not cut:
                # Start the ring at its smallest vertex so identical rings give identical arcs.
                s = min(range(len(pts)), key=lambda i: _key(pts[i]))
                ring = np.vstack([pts[s:], pts[:s], pts[s : s + 1]])
                out.append((addr, [add(ring)]))
                continue
            if closed:
                # Rotate the ring to start at a junction, then close it again.
                s, m = cut[0], len(pts)
                pts = np.vstack([pts[s:], pts[:s], pts[s : s + 1]])
                cut = sorted((c - s) % m for c in cut) + [m]
            else:
                cut = sorted(set([0, len(pts) - 1] + cut))
            seq = [add(pts[a : b + 1]) for a, b in zip(cut[:-1], cut[1:]) if b > a]
            out.append((addr, seq))
        return cls(kind, arcs, out)

    def simplify(self, method: str, tol: float | np.ndarray, extra: list[np.ndarray] | None = None) -> list:
        """Simplified layer. ``tol`` may be a scalar or one tolerance per arc.

        ``extra`` optionally forces additional vertices to be kept, per arc.
        The per-arc keep masks are stored in ``self.keep_masks`` for metrics.
        """
        tols = np.broadcast_to(np.asarray(tol, dtype=float), (len(self.arcs),))
        self.keep_masks = [arc.keep(method, t) for arc, t in zip(self.arcs, tols)]
        if extra is not None:
            self.keep_masks = [k | e for k, e in zip(self.keep_masks, extra)]
        return self.assemble([arc.coords[k] for arc, k in zip(self.arcs, self.keep_masks)])

    def displacement(self) -> np.ndarray:
        """Per-arc maximum displacement of the last simplification.

        Each original vertex is measured against the simplified segment that
        replaced it (between the nearest kept vertices on either side). This is
        the error Douglas-Peucker bounds, and is never smaller than the discrete
        Hausdorff distance between the original and simplified arc.
        """
        out = np.zeros(len(self.arcs))
        for i, (arc, keep) in enumerate(zip(self.arcs, self.keep_masks)):
            n = len(keep)
            if keep.all():
                continue
            pos = np.arange(n)
            prev = np.maximum.accumulate(np.where(keep, pos, 0))
            nxt = np.minimum.accumulate(np.where(keep, pos, n - 1)[::-1])[::-1]
            a, b, p = arc.coords[prev], arc.coords[nxt], arc.coords
            ab = b - a
            denom = np.einsum("ij,ij->i", ab, ab)
            t = np.where(denom > 0, np.einsum("ij,ij->i", p - a, ab) / np.where(denom > 0, denom, 1), 0.0)
            proj = a + np.clip(t, 0, 1)[:, None] * ab
            out[i] = np.hypot(*(p - proj).T).max()
        return out

    @property
    def feature_arcs(self) -> list[np.ndarray]:
        """Arc indices used by each feature."""
        if not hasattr(self, "_feature_arcs"):
            n = 1 + max(addr[0] for addr, _ in self.paths)
            sets: list[set] = [set() for _ in range(n)]
            for addr, seq in self.paths:
                sets[addr[0]].update(i for i, _ in seq)
            self._feature_arcs = [np.array(sorted(s), dtype=int) for s in sets]
        return self._feature_arcs

    def path_coords(self, arc_coords: list[np.ndarray]) -> list[np.ndarray]:
        """Coordinates of every path (ring or line) from per-arc coordinates."""
        out = []
        for _, seq in self.paths:
            parts = []
            for i, rev in seq:
                c = arc_coords[i][::-1] if rev else arc_coords[i]
                parts.append(c if not parts else c[1:])
            out.append(np.vstack(parts))
        return out

    def assemble(self, arc_coords: list[np.ndarray]) -> list:
        """Rebuild features from per-arc coordinates; collapsed rings are dropped."""
        n_features = 1 + max(addr[0] for addr, _ in self.paths)
        features: list = [[] for _ in range(n_features)]
        for addr, seq in self.paths:
            parts = []
            for i, rev in seq:
                c = arc_coords[i][::-1] if rev else arc_coords[i]
                parts.append(c if not parts else c[1:])
            coords = np.vstack(parts)
            if self.kind == "polygon":
                f, p, r = addr
                feat = features[f]
                while len(feat) <= p:
                    feat.append([])
                if r > 0 and not feat[p]:
                    continue  # exterior collapsed: drop its holes too
                if len(np.unique(coords[:-1], axis=0)) < 3:
                    continue  # ring collapsed
                feat[p].append(coords)
            else:
                f, _ = addr
                if len(coords) >= 2:
                    features[f].append(coords)
        if self.kind == "polygon":
            features = [[p for p in feat if p] for feat in features]
        return features

    @property
    def n_vertices(self) -> int:
        return sum(len(a.coords) for a in self.arcs)
