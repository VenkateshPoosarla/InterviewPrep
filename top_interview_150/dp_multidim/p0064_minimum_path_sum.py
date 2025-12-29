"""64. Minimum Path Sum

Link: https://leetcode.com/problems/minimum-path-sum/

Problem:
Given a grid of non-negative numbers, find a path from top-left to bottom-right which
minimizes the sum of all numbers along the path. You can only move right or down.

Approach (DP with rolling row):
dp[c] = min path sum to reach current row, column c.
Transition:
dp[c] = grid[r][c] + min(dp[c] (from top), dp[c-1] (from left))

Complexity:
- Time: O(m*n)
- Space: O(n)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        m, n = len(grid), len(grid[0])
        dp = [0] * n
        dp[0] = grid[0][0]
        for c in range(1, n):
            dp[c] = dp[c - 1] + grid[0][c]
        for r in range(1, m):
            dp[0] += grid[r][0]
            for c in range(1, n):
                dp[c] = grid[r][c] + min(dp[c], dp[c - 1])
        return dp[-1]


def run_tests() -> None:
    sol = Solution()
    assert sol.minPathSum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]) == 7
    assert sol.minPathSum([[1, 2, 3], [4, 5, 6]]) == 12
    assert sol.minPathSum([[0]]) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
