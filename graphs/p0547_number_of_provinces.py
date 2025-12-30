"""547. Number of Provinces

Link: https://leetcode.com/problems/number-of-provinces/

Given an adjacency matrix isConnected, count connected components.
Use DFS/BFS over cities.
"""

from __future__ import annotations

from typing import List


class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        seen = [False] * n

        def dfs(i: int) -> None:
            seen[i] = True
            row = isConnected[i]
            for j in range(n):
                if row[j] == 1 and not seen[j]:
                    dfs(j)

        provinces = 0
        for i in range(n):
            if not seen[i]:
                provinces += 1
                dfs(i)
        return provinces


def run_tests() -> None:
    sol = Solution()

    assert sol.findCircleNum([[1, 1, 0], [1, 1, 0], [0, 0, 1]]) == 2
    assert sol.findCircleNum([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == 3
    assert sol.findCircleNum([[1]]) == 1
    assert sol.findCircleNum([]) == 0


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


