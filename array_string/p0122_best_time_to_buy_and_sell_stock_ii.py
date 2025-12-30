"""122. Best Time to Buy and Sell Stock II

Link: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/

Problem:
You may complete as many transactions as you like (buy then sell), but you can hold
at most one share at a time. Maximize total profit.

Approach (sum of all ascending slopes):
Any increasing segment can be decomposed into day-to-day gains without changing total:
profit = Σ max(0, prices[i] - prices[i-1])

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                profit += prices[i] - prices[i - 1]
        return profit


def run_tests() -> None:
    sol = Solution()
    assert sol.maxProfit([7, 1, 5, 3, 6, 4]) == 7
    assert sol.maxProfit([1, 2, 3, 4, 5]) == 4
    assert sol.maxProfit([7, 6, 4, 3, 1]) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
