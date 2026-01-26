"""63. Unique Paths II

Link: https://leetcode.com/problems/unique-paths-ii/

Problem:
Given an m x n grid with obstacles (1 = obstacle, 0 = empty), return the number of unique
paths from top-left to bottom-right, moving only right or down.

Approach (DP with rolling row):
dp[c] = number of ways to reach current cell in current row.
If obstacle: dp[c] = 0
Else: dp[c] = dp[c] (from top) + dp[c-1] (from left)

Complexity:
- Time: O(m*n)
- Space: O(n)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if not obstacleGrid or not obstacleGrid[0]:
            return 0
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [0] * n
        dp[0] = 0 if obstacleGrid[0][0] == 1 else 1

        for r in range(m):
            for c in range(n):
                if obstacleGrid[r][c] == 1:
                    dp[c] = 0
                else:
                    if c > 0:
                        dp[c] += dp[c - 1]
        return dp[-1]


def run_tests() -> None:
    sol = Solution()
    assert sol.uniquePathsWithObstacles([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) == 2
    assert sol.uniquePathsWithObstacles([[0, 1], [0, 0]]) == 1
    assert sol.uniquePathsWithObstacles([[1]]) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
