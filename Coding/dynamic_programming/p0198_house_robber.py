"""198. House Robber

Link: https://leetcode.com/problems/house-robber/

DP on line of houses:
At house i, either:
- skip it => dp[i-1]
- rob it  => dp[i-2] + nums[i]

Rolling variables:
prev2 = dp[i-2]
prev1 = dp[i-1]
cur = max(prev1, prev2 + nums[i])
"""

from __future__ import annotations

from typing import List


class Solution:
    def rob(self, nums: List[int]) -> int:
        prev2 = 0
        prev1 = 0
        for x in nums:
            cur = max(prev1, prev2 + x)
            prev2, prev1 = prev1, cur
        return prev1


def run_tests() -> None:
    sol = Solution()
    assert sol.rob([1, 2, 3, 1]) == 4
    assert sol.rob([2, 7, 9, 3, 1]) == 12
    assert sol.rob([]) == 0
    assert sol.rob([5]) == 5


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


