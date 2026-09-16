"""Colour-space conversions used when matching pixels to palette entries.

Matching is done in the chosen space, but the reported error is always measured
back in RGB — otherwise the numbers would not be comparable, since the spaces
have different scales (RGB channels run 0-255, HSV components 0-1).
"""

import colorsys

import numpy as np

SPACES = ("rgb", "hsv", "lab")


def convert(space, width, height, palette, px):
    """Convert both the palette and the image into `space`.

    Returns (converted_palette, converted_pixels). For "rgb", the pixels are
    returned as None and the caller reads the original pixel access directly.
    """
    if space == "rgb":
        return palette, None

    if space == "hsv":
        converted_palette = [
            colorsys.rgb_to_hsv(c[0] / 255, c[1] / 255, c[2] / 255) for c in palette
        ]
        converted_pixels = np.zeros((height, width, 3))
        for i in range(width):
            for j in range(height):
                r, g, b = px[i, j]
                converted_pixels[j, i] = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        return converted_palette, converted_pixels

    if space == "lab":
        from skimage import color  # optional dependency, only needed for LAB

        converted_palette = [
            color.rgb2lab(np.array([[c]], dtype=np.uint8))[0][0] for c in palette
        ]
        pixel_array = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(width):
            for j in range(height):
                pixel_array[j, i] = px[i, j]
        return converted_palette, color.rgb2lab(pixel_array)

    raise ValueError(f"unknown colour space {space!r}; pick one of {SPACES}")
