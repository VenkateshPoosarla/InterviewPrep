"""312. Burst Balloons

Link: https://leetcode.com/problems/burst-balloons/

Pattern: Interval DP.

Pad nums with 1 at both ends. Let dp[l][r] be max coins from bursting balloons
strictly between indices l and r (open interval).

Transition:
dp[l][r] = max over k in (l+1..r-1):
    dp[l][k] + nums[l]*nums[k]*nums[r] + dp[k][r]
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        arr = [1] + [int(x) for x in nums] + [1]
        n = len(arr)
        dp = [[0] * n for _ in range(n)]

        for length in range(2, n):  # interval size
            for l in range(0, n - length):
                r = l + length
                best = 0
                for k in range(l + 1, r):
                    coins = dp[l][k] + arr[l] * arr[k] * arr[r] + dp[k][r]
                    if coins > best:
                        best = coins
                dp[l][r] = best

        return dp[0][n - 1]


def run_tests() -> None:
    sol = Solution()
    assert sol.maxCoins([3, 1, 5, 8]) == 167
    assert sol.maxCoins([1, 5]) == 10
    assert sol.maxCoins([]) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


