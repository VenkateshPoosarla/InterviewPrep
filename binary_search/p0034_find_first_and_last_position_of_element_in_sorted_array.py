"""34. Find First and Last Position of Element in Sorted Array

Link: https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/

Problem:
Given a sorted array of integers, find the starting and ending position of a given target.
If target is not found, return [-1, -1].

Approach (two binary searches):
Use lower_bound for:
- left = first index with nums[i] >= target
- right = first index with nums[i] > target, so last occurrence is right-1

Complexity:
- Time: O(log n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def lower(x: int) -> int:
            lo, hi = 0, len(nums)
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        left = lower(target)
        if left == len(nums) or nums[left] != target:
            return [-1, -1]
        right = lower(target + 1) - 1
        return [left, right]


def run_tests() -> None:
    sol = Solution()
    assert sol.searchRange([5, 7, 7, 8, 8, 10], 8) == [3, 4]
    assert sol.searchRange([5, 7, 7, 8, 8, 10], 6) == [-1, -1]
    assert sol.searchRange([], 0) == [-1, -1]
    assert sol.searchRange([1], 1) == [0, 0]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
