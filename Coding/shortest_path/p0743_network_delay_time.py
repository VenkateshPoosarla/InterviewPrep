"""743. Network Delay Time

Link: https://leetcode.com/problems/network-delay-time/

Pattern: Dijkstra (single-source shortest paths on non-negative weights).
"""

from __future__ import annotations

import sys
import heapq
from typing import List, Tuple


class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        g: List[List[Tuple[int, int]]] = [[] for _ in range(n + 1)]
        for u, v, w in times:
            g[int(u)].append((int(v), int(w)))

        INF = 10**18
        dist = [INF] * (n + 1)
        dist[k] = 0
        pq: List[Tuple[int, int]] = [(0, k)]

        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            for v, w in g[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))

        ans = max(dist[1:])
        return -1 if ans >= INF else int(ans)


def run_tests() -> None:
    sol = Solution()
    assert sol.networkDelayTime([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2) == 2
    assert sol.networkDelayTime([[1, 2, 1]], 2, 1) == 1
    assert sol.networkDelayTime([[1, 2, 1]], 2, 2) == -1


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


