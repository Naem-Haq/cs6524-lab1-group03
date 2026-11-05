
import numpy as np
from scipy import ndimage

# Clockwise on screen, starting from west
NEIGHBOURS = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]


def trace_loop(pixels):
    """Moore-neighbour tracing of one connected set of (x, y) pixels, as a closed path."""
    remaining = set(map(tuple, pixels))
    start = min(remaining, key=lambda p: (p[1], p[0]))
    state = (start, 0)  # West of the raster-first pixel is always empty
    path = []
    seen = set()

    while state not in seen:
        seen.add(state)
        (x, y), backtrack = state
        path.append((x, y))
        for turn in range(1, 8):
            d = (backtrack + turn) % 8
            nx, ny = x + NEIGHBOURS[d][0], y + NEIGHBOURS[d][1]
            if (nx, ny) in remaining:
                last_empty = (x + NEIGHBOURS[d - 1][0], y + NEIGHBOURS[d - 1][1])
                offset = (last_empty[0] - nx, last_empty[1] - ny)
                state = ((nx, ny), NEIGHBOURS.index(offset))
                break

    if path[-1] == path[0] and len(path) > 1:
        path.pop()
    return np.array(path, dtype=float)


def stitch_loops(loops):
    """Join closed loops into one by a shortest bridge, walked out and back."""
    merged = loops[0]
    for loop in loops[1:]:
        dists = np.sum((merged[:, None] - loop[None])**2, axis=2)
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        merged = np.vstack((merged[:i + 1], loop[j:], loop[:j + 1], merged[i:]))
    return merged


def calculate(edges, num_samples=500, min_loop_pixels=20):
    """Order boundary pixels into a closed curve and resample it uniformly by arc length."""
    labels, found = ndimage.label(edges, structure=np.ones((3, 3)))
    loops = []
    for label in range(1, found + 1):
        edge_y, edge_x = np.where(labels == label)
        if len(edge_x) >= min_loop_pixels:
            loops.append(trace_loop(np.column_stack((edge_x, edge_y))))
    loops.sort(key=len, reverse=True)
    ordered_pts = stitch_loops(loops)

    closed = np.vstack((ordered_pts, ordered_pts[:1]))
    distances = np.sqrt(np.sum(np.diff(closed, axis=0)**2, axis=1))
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
    target_dists = np.linspace(0, cumulative_dist[-1], num_samples, endpoint=False)

    x_resampled = np.interp(target_dists, cumulative_dist, closed[:, 0])
    y_resampled = np.interp(target_dists, cumulative_dist, closed[:, 1])
    return x_resampled, y_resampled
