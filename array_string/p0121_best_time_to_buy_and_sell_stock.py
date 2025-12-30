"""121. Best Time to Buy and Sell Stock

Link: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

Problem:
Given prices where prices[i] is the price on day i, pick one day to buy and one later
day to sell to maximize profit. Return the maximum profit (or 0 if no profit).

Approach:
Scan once, tracking:
- `min_price`: the cheapest price seen so far (best day to buy up to today)
- `best`: best profit so far using today's price as the sell price

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price = float("inf")
        best = 0
        for p in prices:
            if p < min_price:
                min_price = p
            else:
                best = max(best, p - min_price)
        return best


def run_tests() -> None:
    sol = Solution()
    assert sol.maxProfit([7, 1, 5, 3, 6, 4]) == 5
    assert sol.maxProfit([7, 6, 4, 3, 1]) == 0
    assert sol.maxProfit([1, 2]) == 1
    assert sol.maxProfit([2, 1]) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
