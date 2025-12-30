"""1192. Critical Connections in a Network

Link: https://leetcode.com/problems/critical-connections-in-a-network/

Pattern: Tarjan (bridge-finding) with discovery time + low-link values.

An undirected edge (u, v) is a bridge if low[v] > disc[u] when v is a DFS child of u.
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        g: List[List[int]] = [[] for _ in range(n)]
        for a, b in connections:
            a = int(a)
            b = int(b)
            g[a].append(b)
            g[b].append(a)

        disc = [-1] * n
        low = [0] * n
        time = 0
        bridges: List[List[int]] = []

        def dfs(u: int, parent: int) -> None:
            nonlocal time
            disc[u] = time
            low[u] = time
            time += 1

            for v in g[u]:
                if v == parent:
                    continue
                if disc[v] == -1:
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        bridges.append([u, v])
                else:
                    low[u] = min(low[u], disc[v])

        for i in range(n):
            if disc[i] == -1:
                dfs(i, -1)

        return bridges


def run_tests() -> None:
    sol = Solution()

    out = sol.criticalConnections(4, [[0, 1], [1, 2], [2, 0], [1, 3]])
    assert sorted(map(sorted, out)) == [[1, 3]]

    out = sol.criticalConnections(2, [[0, 1]])
    assert sorted(map(sorted, out)) == [[0, 1]]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


