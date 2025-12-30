"""714. Best Time to Buy and Sell Stock with Transaction Fee

Link: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/

DP with two states:
- cash: max profit when not holding a stock
- hold: max profit when holding a stock

Transitions for price p:
  new_cash = max(cash, hold + p - fee)   # sell or keep cash
  new_hold = max(hold, cash - p)         # buy or keep hold
"""

from __future__ import annotations

from typing import List


class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        cash = 0
        hold = -10**18
        for p in prices:
            new_cash = max(cash, hold + p - fee)
            new_hold = max(hold, cash - p)
            cash, hold = new_cash, new_hold
        return cash


def run_tests() -> None:
    sol = Solution()
    assert sol.maxProfit([1, 3, 2, 8, 4, 9], 2) == 8
    assert sol.maxProfit([1, 3, 7, 5, 10, 3], 3) == 6
    assert sol.maxProfit([], 2) == 0


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


