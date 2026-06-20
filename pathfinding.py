"""A* pathfinding on a binary traversability mask.

Public API:
    astar(mask, start, end) -> list[(row, col)] | None
"""

from __future__ import annotations

import heapq
from typing import List, Optional, Tuple

import numpy as np

# (delta_row, delta_col, move_cost)
_MOVES: list[tuple[int, int, float]] = [
    (-1,  0, 1.0),
    ( 1,  0, 1.0),
    ( 0, -1, 1.0),
    ( 0,  1, 1.0),
    (-1, -1, 1.414),
    (-1,  1, 1.414),
    ( 1, -1, 1.414),
    ( 1,  1, 1.414),
]


def _snap_to_traversable(mask: np.ndarray, row: int, col: int) -> tuple[int, int]:
    """Return the nearest traversable pixel to (row, col)."""
    if mask[row, col] == 1:
        return row, col
    rows, cols = np.where(mask == 1)
    if len(rows) == 0:
        return row, col  # mask is all obstacle — caller handles None path
    dists = (rows - row) ** 2 + (cols - col) ** 2
    idx = int(np.argmin(dists))
    return int(rows[idx]), int(cols[idx])


def astar(
    mask: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    max_nodes: int = 500_000,
) -> Optional[List[Tuple[int, int]]]:
    """Find the shortest traversable path using A*.

    Args:
        mask:      (H, W) uint8 array — 1=traversable, 0=obstacle.
        start:     (row, col) of the start pixel.
        end:       (row, col) of the end pixel.
        max_nodes: abort after visiting this many nodes (prevents hangs on
                   near-impossible paths on large masks).

    Returns:
        Ordered list of (row, col) from start to end, or None if unreachable.
    """
    H, W = mask.shape

    # Clamp coordinates to grid bounds
    start = (max(0, min(H - 1, start[0])), max(0, min(W - 1, start[1])))
    end   = (max(0, min(H - 1, end[0])),   max(0, min(W - 1, end[1])))

    # Snap to nearest traversable pixel if user clicked an obstacle
    start = _snap_to_traversable(mask, *start)
    end   = _snap_to_traversable(mask, *end)

    if start == end:
        return [start]

    def h(r: int, c: int) -> float:
        return ((r - end[0]) ** 2 + (c - end[1]) ** 2) ** 0.5

    # heap entries: (f_score, g_score, (row, col))
    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (h(*start), 0.0, start))

    g: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    visited = 0

    while open_heap:
        _, g_cur, cur = heapq.heappop(open_heap)
        visited += 1

        if cur == end:
            path: list[tuple[int, int]] = []
            node = end
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            return path[::-1]

        # Skip stale heap entries
        if g_cur > g.get(cur, float("inf")):
            continue

        if visited > max_nodes:
            return None

        r, c = cur
        for dr, dc, cost in _MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] == 1:
                ng = g_cur + cost
                nb = (nr, nc)
                if ng < g.get(nb, float("inf")):
                    g[nb] = ng
                    came_from[nb] = cur
                    heapq.heappush(open_heap, (ng + h(nr, nc), ng, nb))

    return None
