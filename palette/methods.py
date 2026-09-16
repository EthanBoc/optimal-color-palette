"""The four palette-selection methods, from the most naive to the most refined.

Every method takes the image's pixel access and a target palette size k, and
returns a list of k RGB colours.
"""

import random

from .core import colour_ranking, distance, sorted_colours


def greedy(width, height, px, k):
    """Baseline: keep the k most frequent colours in the image.

    Works well when k is small and the image has a few dominant colours, but
    ignores everything rare — subtle gradients collapse onto the dominant hues.
    """
    return sorted_colours(width, height, px)[:k]


def frequency_intervals(width, height, px, k):
    """Split the frequency-sorted colour list into k intervals, sample each one.

    Same ranking as `greedy`, but instead of taking the top k, the list is cut
    into k regular slices so that rare colours stay represented. Noticeably
    better spread than greedy for the same cost.
    """
    colours = sorted_colours(width, height, px)
    step = max(1, len(colours) // k)
    return colours[::step][:k]


def _average_colour(box):
    """Mean colour of a box. Linear in the box size."""
    r = g = b = 0
    for colour in box:
        r += colour[0]
        g += colour[1]
        b += colour[2]
    n = len(box)
    return (r // n, g // n, b // n)


def _widest_channel(box):
    """Return (spread, channel_index) for the channel with the widest range."""
    r_max = g_max = b_max = 0
    r_min = g_min = b_min = 255
    for r, g, b in box:
        r_max, r_min = max(r_max, r), min(r_min, r)
        g_max, g_min = max(g_max, g), min(g_min, g)
        b_max, b_min = max(b_max, b), min(b_min, b)

    spreads = [(r_max - r_min, 0), (g_max - g_min, 1), (b_max - b_min, 2)]
    return max(spreads)


def _split_box(boxes, box_index, channel):
    """Split one box into two halves of equal size along `channel`.

    O(n log n) in the box size, because of the sort.
    """
    box = boxes.pop(box_index)
    box.sort(key=lambda colour: colour[channel])
    half = len(box) // 2
    boxes.append(box[:half])
    boxes.append(box[half:])
    return boxes


def median_cut(width, height, px, k):
    """Median cut: recursively split the 3D RGB cube into k boxes.

    Every pixel starts in one box. At each step the box with the widest spread
    on any single channel is split in two at its median along that channel.
    The palette is the mean colour of each final box, so the palette adapts to
    how the colours are actually distributed rather than to how often they occur.
    """
    boxes = [[px[i, j] for i in range(width) for j in range(height)]]

    while len(boxes) < k:
        best_spread = -1
        best_index = 0
        best_channel = 0
        for index, box in enumerate(boxes):
            spread, channel = _widest_channel(box)
            if spread > best_spread:
                best_spread, best_index, best_channel = spread, index, channel

        if best_spread <= 0:  # every box is uniform: nothing left to split
            break
        boxes = _split_box(boxes, best_index, best_channel)

    return [_average_colour(box) for box in boxes]


def _centroid(points):
    n = len(points[0])
    total = [0] * n
    for point in points:
        for i in range(n):
            total[i] += point[i]
    return tuple(value / len(points) for value in total)


def _nearest_centroid_index(point, centroids):
    best_index = 0
    best_distance = float("inf")
    for index, centroid in enumerate(centroids):
        d = distance(point, centroid)
        if d < best_distance:
            best_distance, best_index = d, index
    return best_index


def k_means(points, k, tolerance=1e-3, seed=None):
    """Plain Lloyd's algorithm, written from scratch (no scikit-learn).

    Iterates until every centroid moves less than `tolerance`. Empty clusters
    are re-seeded with a random point so the palette always ends up with k
    colours.
    """
    if seed is not None:
        random.seed(seed)

    centroids = random.sample(points, k)
    clusters = [[] for _ in range(k)]
    for point in points:
        clusters[_nearest_centroid_index(point, centroids)].append(point)

    while True:
        previous = centroids
        centroids = [
            _centroid(cluster) if cluster else random.choice(points)
            for cluster in clusters
        ]

        clusters = [[] for _ in range(k)]
        for point in points:
            clusters[_nearest_centroid_index(point, centroids)].append(point)

        if all(distance(previous[i], centroids[i]) < tolerance for i in range(k)):
            return clusters, centroids


def k_means_palette(width, height, px, k, seed=None):
    """K-means over the image's *unique* colours.

    Note this clusters unique colours, not pixels, so a colour covering half
    the image weighs the same as one used by a single pixel. That is a
    deliberate difference from `median_cut`, and it shows in the results.
    """
    colours = list(colour_ranking(width, height, px))
    clusters, _ = k_means(colours, k, seed=seed)
    return [_average_colour(cluster) for cluster in clusters if cluster]


METHODS = {
    "greedy": greedy,
    "intervals": frequency_intervals,
    "median-cut": median_cut,
    "k-means": k_means_palette,
}


def build_palette(name, width, height, px, k, **kwargs):
    """Dispatch to one of the four methods by name."""
    if name not in METHODS:
        raise ValueError(f"unknown method {name!r}; pick one of {sorted(METHODS)}")
    if name == "k-means":
        return METHODS[name](width, height, px, k, **kwargs)
    return METHODS[name](width, height, px, k)
