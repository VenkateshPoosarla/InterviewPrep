"""746. Min Cost Climbing Stairs

Link: https://leetcode.com/problems/min-cost-climbing-stairs/

You can start at step 0 or 1, and pay cost[i] when stepping on i.
DP:
dp[i] = min cost to reach step i (before paying for i).
Answer is min cost to reach "top" at index n.

Rolling:
dp0 = min cost to reach i-2
dp1 = min cost to reach i-1
for i in 2..n:
  dp = min(dp1 + cost[i-1], dp0 + cost[i-2])
"""

from __future__ import annotations

from typing import List


class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        dp0, dp1 = 0, 0
        for i in range(2, n + 1):
            dp = min(dp1 + cost[i - 1], dp0 + cost[i - 2])
            dp0, dp1 = dp1, dp
        return dp1


def run_tests() -> None:
    sol = Solution()
    assert sol.minCostClimbingStairs([10, 15, 20]) == 15
    assert sol.minCostClimbingStairs([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]) == 6
    assert sol.minCostClimbingStairs([0, 0]) == 0


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


