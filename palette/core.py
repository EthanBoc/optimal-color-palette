"""Core image utilities: pixel access, colour counting, distance, convolution."""

from math import sqrt

from PIL import Image

# Gaussian blur kernels (unnormalised; the convolution divides by their sum)
GAUSS3 = [
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
]

GAUSS7 = [
    [1, 1, 2, 2, 2, 1, 1],
    [1, 2, 2, 4, 2, 2, 1],
    [2, 2, 4, 8, 4, 2, 2],
    [2, 4, 8, 16, 8, 4, 2],
    [2, 2, 4, 8, 4, 2, 2],
    [1, 2, 2, 4, 2, 2, 1],
    [1, 1, 2, 2, 2, 1, 1],
]

KERNELS = {"gauss3": GAUSS3, "gauss7": GAUSS7}


def load_image(path):
    """Open an image as RGB and return (image, pixel_access, width, height)."""
    image = Image.open(path).convert("RGB")
    return image, image.load(), image.width, image.height


def colour_ranking(width, height, px):
    """Count how many times each exact RGB colour appears in the image."""
    colours = {}
    for i in range(width):
        for j in range(height):
            pixel = px[i, j]
            colours[pixel] = colours.get(pixel, 0) + 1
    return colours


def sorted_colours(width, height, px):
    """Unique colours, most frequent first."""
    ranking = colour_ranking(width, height, px)
    return sorted(ranking, key=ranking.get, reverse=True)


def distance(c1, c2):
    """Euclidean distance between two colours, in whatever space they live in."""
    return sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2 + (c2[2] - c1[2]) ** 2)


def matrix_sum(matrix):
    return sum(sum(row) for row in matrix)


def convolution(px, width, height, kernel):
    """Apply a convolution kernel in place.

    Pixels closer to the border than the kernel size are left untouched, so the
    kernel never reads outside the image. Complexity: O(W * H * n^2) for an
    n x n kernel.
    """
    n = len(kernel)
    total = matrix_sum(kernel) or 1  # a zero-sum kernel (edge detection) is left unnormalised

    for i in range(n, width - n):
        for j in range(n, height - n):
            acc_r = acc_g = acc_b = 0
            for w in range(-(n // 2), n // 2 + 1):
                for h in range(-(n // 2), n // 2 + 1):
                    r, g, b = px[i + w, j + h]
                    weight = kernel[w + n // 2][h + n // 2]
                    acc_r += r * weight
                    acc_g += g * weight
                    acc_b += b * weight
            px[i, j] = (int(acc_r / total), int(acc_g / total), int(acc_b / total))
