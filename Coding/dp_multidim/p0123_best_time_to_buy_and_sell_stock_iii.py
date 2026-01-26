"""123. Best Time to Buy and Sell Stock III

Link: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/

Problem:
Given prices where prices[i] is the price on day i, return the max profit with at most
two transactions. You may not hold multiple shares at once.

Approach (state machine DP):
Maintain the best value after each state:
- buy1: max profit after first buy (negative)
- sell1: max profit after first sell
- buy2: max profit after second buy
- sell2: max profit after second sell

Update per price p:
buy1 = max(buy1, -p)
sell1 = max(sell1, buy1 + p)
buy2 = max(buy2, sell1 - p)
sell2 = max(sell2, buy2 + p)

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        buy1 = float("-inf")
        sell1 = 0
        buy2 = float("-inf")
        sell2 = 0

        for p in prices:
            buy1 = max(buy1, -p)
            sell1 = max(sell1, buy1 + p)
            buy2 = max(buy2, sell1 - p)
            sell2 = max(sell2, buy2 + p)
        return sell2


def run_tests() -> None:
    sol = Solution()
    assert sol.maxProfit([3, 3, 5, 0, 0, 3, 1, 4]) == 6
    assert sol.maxProfit([1, 2, 3, 4, 5]) == 4
    assert sol.maxProfit([7, 6, 4, 3, 1]) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
