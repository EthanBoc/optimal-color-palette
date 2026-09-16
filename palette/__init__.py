"""Automatic optimal colour-palette selection for images."""

from .core import GAUSS3, GAUSS7, colour_ranking, convolution, distance, load_image
from .methods import METHODS, build_palette
from .quantize import benchmark, pil_quantize, quantize

__all__ = [
    "GAUSS3", "GAUSS7", "colour_ranking", "convolution", "distance", "load_image",
    "METHODS", "build_palette", "benchmark", "pil_quantize", "quantize",
]
