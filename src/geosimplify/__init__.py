"""geosimplify - topology-aware line simplification for web maps, with a benchmark."""

from .algorithms import dp_significance, radial_distance, simplify_line, vw_significance
from .core import MODES, Layer, Simplifier
from .encoding import layer_size
from .topology import ArcGraph, resolution, to_mercator

__all__ = [
    "MODES", "ArcGraph", "Layer", "Simplifier", "dp_significance", "layer_size", "radial_distance",
    "resolution", "simplify_line", "to_mercator", "vw_significance",
]
__version__ = "0.1.0"
