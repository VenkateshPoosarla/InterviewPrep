"""834. Sum of Distances in Tree

Link: https://leetcode.com/problems/sum-of-distances-in-tree/

Pattern: Tree DP + rerooting.

1) Postorder from an arbitrary root (0):
   - size[u] = subtree size
   - ans0[u] = sum of distances from u to nodes in its subtree (when rooted at 0)
2) Reroot:
   - ans[v] = ans[u] - size[v] + (n - size[v]) for edge u-v where v is child of u.
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def sumOfDistancesInTree(self, n: int, edges: List[List[int]]) -> List[int]:
        g: List[List[int]] = [[] for _ in range(n)]
        for a, b in edges:
            a = int(a)
            b = int(b)
            g[a].append(b)
            g[b].append(a)

        size = [1] * n
        ans = [0] * n

        def post(u: int, p: int) -> None:
            for v in g[u]:
                if v == p:
                    continue
                post(v, u)
                size[u] += size[v]
                ans[u] += ans[v] + size[v]

        def pre(u: int, p: int) -> None:
            for v in g[u]:
                if v == p:
                    continue
                ans[v] = ans[u] - size[v] + (n - size[v])
                pre(v, u)

        post(0, -1)
        pre(0, -1)
        return ans


def run_tests() -> None:
    sol = Solution()
    assert sol.sumOfDistancesInTree(6, [[0, 1], [0, 2], [2, 3], [2, 4], [2, 5]]) == [8, 12, 6, 10, 10, 10]
    assert sol.sumOfDistancesInTree(1, []) == [0]
    assert sol.sumOfDistancesInTree(2, [[1, 0]]) == [1, 1]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


