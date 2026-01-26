"""847. Shortest Path Visiting All Nodes

Link: https://leetcode.com/problems/shortest-path-visiting-all-nodes/

Pattern: BFS over (node, visited_mask).

State graph has n * 2^n states (n <= 12). Use multi-source BFS starting from each node.
"""

from __future__ import annotations

import sys
from collections import deque
from typing import Deque, List, Tuple


class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        n = len(graph)
        if n <= 1:
            return 0

        full = (1 << n) - 1
        dist = [[-1] * (1 << n) for _ in range(n)]
        q: Deque[Tuple[int, int]] = deque()

        for i in range(n):
            m = 1 << i
            dist[i][m] = 0
            q.append((i, m))

        while q:
            u, mask = q.popleft()
            d = dist[u][mask]
            if mask == full:
                return d
            for v in graph[u]:
                nmask = mask | (1 << v)
                if dist[v][nmask] == -1:
                    dist[v][nmask] = d + 1
                    q.append((v, nmask))

        return -1


def run_tests() -> None:
    sol = Solution()
    assert sol.shortestPathLength([[1, 2, 3], [0], [0], [0]]) == 4
    assert sol.shortestPathLength([[1], [0, 2, 4], [1, 3, 4], [2], [1, 2]]) == 4
    assert sol.shortestPathLength([[]]) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


