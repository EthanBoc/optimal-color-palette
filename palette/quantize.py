"""Recolour an image with a palette, and measure how much was lost doing it."""

import time

import numpy as np
from PIL import Image

from .colour_spaces import convert
from .core import KERNELS, convolution, distance, load_image
from .methods import build_palette


def quantize(path, k, method="greedy", space="rgb", blur=None, seed=None):
    """Recolour `path` with a k-colour palette.

    Returns (recoloured_image, palette, heatmap, mean_error).

    - `method`  one of greedy / intervals / median-cut / k-means
    - `space`   matching space: rgb / hsv / lab
    - `blur`    optional "gauss3" or "gauss7" pre-processing pass

    The error is always the RGB distance between the *original* pixel and the
    colour it was replaced with, so results stay comparable across spaces and
    across blur settings.
    """
    original, original_px, width, height = load_image(path)
    working, working_px, _, _ = load_image(path)

    if blur is not None:
        convolution(working_px, width, height, KERNELS[blur])

    palette = build_palette(
        method, width, height, working_px, k, **({"seed": seed} if method == "k-means" else {})
    )
    converted_palette, converted_pixels = convert(space, width, height, palette, working_px)

    result = Image.new("RGB", (width, height))
    result_px = result.load()
    heatmap = np.zeros((height, width))
    total_error = 0.0

    for i in range(width):
        for j in range(height):
            pixel = converted_pixels[j, i] if converted_pixels is not None else working_px[i, j]

            best_colour = palette[0]
            best_distance = float("inf")
            for index, candidate in enumerate(converted_palette):
                d = distance(pixel, candidate)
                if d < best_distance:
                    best_distance, best_colour = d, palette[index]

            error = distance(original_px[i, j], best_colour)
            heatmap[j, i] = error
            total_error += error
            result_px[i, j] = best_colour

    return result, palette, heatmap, total_error / (width * height)


def pil_quantize(path, k):
    """Pillow's own quantiser, used as a reference point."""
    original, original_px, width, height = load_image(path)

    start = time.time()
    result = original.quantize(colors=k).convert("RGB")
    elapsed = time.time() - start

    result_px = result.load()
    heatmap = np.zeros((height, width))
    total_error = 0.0
    for i in range(width):
        for j in range(height):
            error = distance(original_px[i, j], result_px[i, j])
            heatmap[j, i] = error
            total_error += error

    return result, heatmap, total_error / (width * height), elapsed


def benchmark(path, k, methods=None, spaces=("rgb",), blurs=(None,), seed=0):
    """Run several configurations and collect error + runtime for each."""
    methods = methods or ["greedy", "intervals", "median-cut", "k-means"]
    rows = []
    for space in spaces:
        for blur in blurs:
            for method in methods:
                start = time.time()
                _, _, _, error = quantize(path, k, method, space, blur, seed)
                rows.append(
                    {
                        "method": method,
                        "space": space,
                        "blur": blur or "none",
                        "error": round(error, 2),
                        "seconds": round(time.time() - start, 2),
                    }
                )
    return rows
