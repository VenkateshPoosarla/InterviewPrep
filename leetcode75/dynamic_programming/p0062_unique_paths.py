"""62. Unique Paths

Link: https://leetcode.com/problems/unique-paths/

Robot moves only right/down from top-left to bottom-right on an m x n grid.

DP:
dp[c] = number of ways to reach current row, column c.
Initialize first row to 1s. For each row, dp[c] += dp[c-1].
"""

from __future__ import annotations


class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1] * n
        for _ in range(1, m):
            for c in range(1, n):
                dp[c] += dp[c - 1]
        return dp[-1] if dp else 0


def run_tests() -> None:
    sol = Solution()
    assert sol.uniquePaths(3, 7) == 28
    assert sol.uniquePaths(3, 2) == 3
    assert sol.uniquePaths(1, 1) == 1


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


