"""2290. Minimum Obstacle Removal to Reach Corner

Link: https://leetcode.com/problems/minimum-obstacle-removal-to-reach-corner/

Pattern: 0-1 BFS (edges have weight 0 or 1).

Moving into a cell costs grid[r][c] (0 if empty, 1 if obstacle removed).
Use deque and pushleft for 0-cost edges, pushright for 1-cost edges.
"""

from __future__ import annotations

import sys
from collections import deque
from typing import Deque, List, Tuple


class Solution:
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0]) if m else 0
        if m == 0 or n == 0:
            return 0

        INF = 10**18
        dist = [[INF] * n for _ in range(m)]
        dist[0][0] = 0
        dq: Deque[Tuple[int, int]] = deque([(0, 0)])

        while dq:
            r, c = dq.popleft()
            d = dist[r][c]
            if r == m - 1 and c == n - 1:
                return int(d)

            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < m and 0 <= nc < n):
                    continue
                w = 1 if grid[nr][nc] else 0
                nd = d + w
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    if w == 0:
                        dq.appendleft((nr, nc))
                    else:
                        dq.append((nr, nc))

        return int(dist[m - 1][n - 1])


def run_tests() -> None:
    sol = Solution()
    assert sol.minimumObstacles([[0, 1, 1], [1, 1, 0], [1, 1, 0]]) == 2
    assert sol.minimumObstacles([[0, 1, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 0]]) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


