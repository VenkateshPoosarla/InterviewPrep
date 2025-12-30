"""153. Find Minimum in Rotated Sorted Array

Link: https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/

Binary search on rotated sorted array with distinct values.
We compare mid with rightmost to decide which half contains the minimum.
"""

from __future__ import annotations

from typing import List


class Solution:
    def findMin(self, nums: List[int]) -> int:
        lo, hi = 0, len(nums) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] > nums[hi]:
                lo = mid + 1
            else:
                hi = mid
        return nums[lo]


def run_tests() -> None:
    sol = Solution()
    assert sol.findMin([3, 4, 5, 1, 2]) == 1
    assert sol.findMin([4, 5, 6, 7, 0, 1, 2]) == 0
    assert sol.findMin([11, 13, 15, 17]) == 11


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


