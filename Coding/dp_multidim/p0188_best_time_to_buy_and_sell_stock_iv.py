"""188. Best Time to Buy and Sell Stock IV

Link: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/

Problem:
Return the maximum profit you can achieve with at most k transactions.
You may not hold multiple shares at once.

Approach:
- If k >= n//2, it's equivalent to unlimited transactions: sum all positive deltas.
- Otherwise, DP over transactions using two arrays:
  buy[t]  = best profit after buying the (t+1)-th stock (t in [0..k-1])
  sell[t] = best profit after selling the (t+1)-th stock

Update per price p:
buy[0] = max(buy[0], -p)
sell[0] = max(sell[0], buy[0] + p)
for t=1..k-1:
  buy[t] = max(buy[t], sell[t-1] - p)
  sell[t] = max(sell[t], buy[t] + p)

Complexity:
- Time: O(n*k)
- Space: O(k)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        if k <= 0 or n < 2:
            return 0

        # unlimited transactions optimization
        if k >= n // 2:
            profit = 0
            for i in range(1, n):
                if prices[i] > prices[i - 1]:
                    profit += prices[i] - prices[i - 1]
            return profit

        buy = [float("-inf")] * k
        sell = [0] * k

        for p in prices:
            buy[0] = max(buy[0], -p)
            sell[0] = max(sell[0], buy[0] + p)
            for t in range(1, k):
                buy[t] = max(buy[t], sell[t - 1] - p)
                sell[t] = max(sell[t], buy[t] + p)
        return sell[-1]


def run_tests() -> None:
    sol = Solution()
    assert sol.maxProfit(2, [2, 4, 1]) == 2
    assert sol.maxProfit(2, [3, 2, 6, 5, 0, 3]) == 7
    assert sol.maxProfit(100, [1, 2, 3, 4, 5]) == 4  # unlimited path
    assert sol.maxProfit(0, [1, 2]) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
