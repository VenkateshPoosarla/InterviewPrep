"""209. Minimum Size Subarray Sum

Link: https://leetcode.com/problems/minimum-size-subarray-sum/

Problem:
Given an array of positive integers `nums` and a positive integer `target`, return the
minimal length of a contiguous subarray whose sum is at least `target`. If there is no
such subarray, return 0.

Approach (sliding window):
Because all numbers are positive, expanding the right end only increases the sum, and
shrinking the left end only decreases it.
- Expand `r`, add nums[r] to sum.
- While sum >= target, update best and shrink from left.

Complexity:
- Time: O(n) (each index enters/leaves the window once)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        best = float("inf")
        s = 0
        l = 0
        for r, x in enumerate(nums):
            s += x
            while s >= target:
                best = min(best, r - l + 1)
                s -= nums[l]
                l += 1
        return 0 if best == float("inf") else int(best)


def run_tests() -> None:
    sol = Solution()
    assert sol.minSubArrayLen(7, [2, 3, 1, 2, 4, 3]) == 2
    assert sol.minSubArrayLen(4, [1, 4, 4]) == 1
    assert sol.minSubArrayLen(11, [1, 1, 1, 1, 1, 1, 1, 1]) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
