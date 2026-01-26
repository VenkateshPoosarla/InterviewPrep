"""1857. Largest Color Value in a Directed Graph

Link: https://leetcode.com/problems/largest-color-value-in-a-directed-graph/

Pattern: Topological sort + DP on DAG.

Let dp[v][c] = maximum count of color c on any path ending at node v.
Initialize dp[v][color(v)] = 1, then relax along edges in topological order.
If there is a cycle, topological processing won't visit all nodes -> return -1.
"""

from __future__ import annotations

import sys
from collections import deque
from typing import Deque, List


class Solution:
    def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
        n = len(colors)
        g: List[List[int]] = [[] for _ in range(n)]
        indeg = [0] * n
        for u, v in edges:
            g[u].append(v)
            indeg[v] += 1

        dp = [[0] * 26 for _ in range(n)]
        for i, ch in enumerate(colors):
            dp[i][ord(ch) - ord("a")] = 1

        q: Deque[int] = deque(i for i in range(n) if indeg[i] == 0)
        seen = 0
        best = 0

        while q:
            u = q.popleft()
            seen += 1
            best = max(best, max(dp[u]))
            for v in g[u]:
                cv = ord(colors[v]) - ord("a")
                row_u = dp[u]
                row_v = dp[v]
                for c in range(26):
                    cand = row_u[c] + (1 if c == cv else 0)
                    if cand > row_v[c]:
                        row_v[c] = cand
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        return -1 if seen != n else best


def run_tests() -> None:
    sol = Solution()

    assert sol.largestPathValue("abaca", [[0, 1], [0, 2], [2, 3], [3, 4]]) == 3
    assert sol.largestPathValue("a", [[0, 0]]) == -1
    assert sol.largestPathValue("abc", []) == 1


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


