"""322. Coin Change

Link: https://leetcode.com/problems/coin-change/

Problem:
Given coins of different denominations and a total amount, return the fewest number
of coins needed to make up that amount. If not possible, return -1.

Approach (bottom-up DP):
dp[a] = fewest coins to make amount a.
dp[0] = 0.
For each amount a, try each coin:
  dp[a] = min(dp[a], dp[a-coin] + 1)

Complexity:
- Time: O(amount * len(coins))
- Space: O(amount)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        INF = amount + 1
        dp = [INF] * (amount + 1)
        dp[0] = 0

        for a in range(1, amount + 1):
            for c in coins:
                if c <= a:
                    dp[a] = min(dp[a], dp[a - c] + 1)

        return -1 if dp[amount] == INF else dp[amount]


def run_tests() -> None:
    sol = Solution()
    assert sol.coinChange([1, 2, 5], 11) == 3
    assert sol.coinChange([2], 3) == -1
    assert sol.coinChange([1], 0) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
