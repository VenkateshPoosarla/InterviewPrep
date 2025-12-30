"""53. Maximum Subarray

Link: https://leetcode.com/problems/maximum-subarray/

Problem:
Given an integer array, find the contiguous subarray with the largest sum and return its sum.

Approach (Kadane's algorithm):
Maintain:
- cur = best subarray sum ending at current index
- best = best subarray sum seen so far
Transition:
  cur = max(x, cur + x)
  best = max(best, cur)

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        cur = nums[0]
        best = nums[0]
        for x in nums[1:]:
            cur = max(x, cur + x)
            best = max(best, cur)
        return best


def run_tests() -> None:
    sol = Solution()
    assert sol.maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
    assert sol.maxSubArray([1]) == 1
    assert sol.maxSubArray([5, 4, -1, 7, 8]) == 23


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
