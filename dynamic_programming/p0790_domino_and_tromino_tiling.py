"""790. Domino and Tromino Tiling

Link: https://leetcode.com/problems/domino-and-tromino-tiling/

Let dp[n] be number of ways to fully tile a 2xn board.
Known recurrence:
  dp[0]=1, dp[1]=1, dp[2]=2
  dp[n] = 2*dp[n-1] + dp[n-3]   for n>=3
All results mod 1e9+7.
"""

from __future__ import annotations


class Solution:
    def numTilings(self, n: int) -> int:
        MOD = 1_000_000_007
        if n == 0:
            return 1
        if n == 1:
            return 1
        if n == 2:
            return 2

        dp0, dp1, dp2 = 1, 1, 2  # dp[n-3], dp[n-2], dp[n-1] when iterating
        for _ in range(3, n + 1):
            dp = (2 * dp2 + dp0) % MOD
            dp0, dp1, dp2 = dp1, dp2, dp
        return dp2


def run_tests() -> None:
    sol = Solution()
    assert sol.numTilings(1) == 1
    assert sol.numTilings(2) == 2
    assert sol.numTilings(3) == 5
    assert sol.numTilings(4) == 11


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


