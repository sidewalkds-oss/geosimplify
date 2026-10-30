"""Vector-tile geometry size, following the Mapbox Vector Tile (MVT 2.1) encoding.

MVT stores geometry as command integers (MoveTo, LineTo, ClosePath) followed by
zig-zag encoded coordinate deltas, all as protobuf varints, on a grid of
``extent`` units per tile. This module counts those bytes on the global tile
grid at a zoom level. It does not clip features to tile boundaries and omits
feature, layer and attribute overhead, so it measures how simplification
changes geometry payload, not a deployable tile size.
"""

from __future__ import annotations

import zlib

import numpy as np

from .topology import EARTH_RADIUS

MOVE_TO, LINE_TO, CLOSE_PATH = 1, 2, 7


def _zigzag(v: np.ndarray) -> np.ndarray:
    return (v << 1) ^ (v >> 63)


def _varint_bytes(values: np.ndarray) -> bytes:
    out = bytearray()
    for v in values.tolist():
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
    return bytes(out)


def quantize(coords: np.ndarray, zoom: int, extent: int = 4096) -> np.ndarray:
    """Web Mercator metres to integer tile-grid units at ``zoom``."""
    world = 2 * np.pi * EARTH_RADIUS
    units = extent * 2**zoom
    x = (coords[:, 0] + world / 2) / world * units
    y = (world / 2 - coords[:, 1]) / world * units
    return np.round(np.column_stack([x, y])).astype(np.int64)


def _command(cmd: int, count: int) -> int:
    return (cmd & 0x7) | (count << 3)


def encode_feature(kind: str, feature: list, zoom: int, extent: int = 4096) -> bytes:
    """MVT command stream for one feature on the global grid at ``zoom``."""
    ints: list[int] = []
    cursor = np.zeros(2, dtype=np.int64)
    paths = [r for p in feature for r in p] if kind == "polygon" else feature
    for path in paths:
        q = quantize(path, zoom, extent)
        if kind == "polygon":
            q = q[:-1]  # the closing vertex is implied by ClosePath
        q = q[np.r_[True, np.any(q[1:] != q[:-1], axis=1)]]  # repeated grid points are dropped
        if len(q) < (3 if kind == "polygon" else 2):
            continue  # degenerate at this resolution
        deltas = np.diff(np.vstack([cursor, q]), axis=0)
        cursor = q[-1]
        z = _zigzag(deltas.ravel()).tolist()
        ints += [_command(MOVE_TO, 1), z[0], z[1], _command(LINE_TO, len(q) - 1), *z[2:]]
        if kind == "polygon":
            ints.append(_command(CLOSE_PATH, 1))
    return _varint_bytes(np.asarray(ints, dtype=np.int64)) if ints else b""


def layer_size(kind: str, features: list, zoom: int) -> dict:
    """Total geometry bytes, raw and zlib-compressed, for a layer at ``zoom``."""
    blobs = [encode_feature(kind, f, zoom) for f in features]
    raw = b"".join(blobs)
    return {"bytes": len(raw), "bytes_zlib": len(zlib.compress(raw, 6))}
