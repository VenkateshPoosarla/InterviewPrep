"""200. Number of Islands

Link: https://leetcode.com/problems/number-of-islands/

Problem:
Given a 2D grid of '1's (land) and '0's (water), return the number of islands.
An island is surrounded by water and is formed by connecting adjacent lands
horizontally or vertically.

Approach (DFS flood fill):
For each cell that is '1', increment count and DFS to mark all connected land as visited.
We can mark visited by mutating grid cells from '1' -> '0' (allowed by LeetCode).

Complexity:
- Time: O(m*n)
- Space: O(m*n) worst-case recursion (or stack), but typically O(m*n) upper bound
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0
        m, n = len(grid), len(grid[0])

        sys.setrecursionlimit(10**6)

        def dfs(r: int, c: int) -> None:
            if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != "1":
                return
            grid[r][c] = "0"
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)

        islands = 0
        for r in range(m):
            for c in range(n):
                if grid[r][c] == "1":
                    islands += 1
                    dfs(r, c)
        return islands


def run_tests() -> None:
    sol = Solution()
    grid = [
        ["1", "1", "1", "1", "0"],
        ["1", "1", "0", "1", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
    ]
    assert sol.numIslands([row[:] for row in grid]) == 1

    grid = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"],
    ]
    assert sol.numIslands([row[:] for row in grid]) == 3


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
