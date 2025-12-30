"""33. Search in Rotated Sorted Array

Link: https://leetcode.com/problems/search-in-rotated-sorted-array/

Binary search with rotation:
At each step, one side (lo..mid or mid..hi) is sorted.
Use that to decide which side to keep.
"""

from __future__ import annotations

from typing import List


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        lo, hi = 0, len(nums) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if nums[mid] == target:
                return mid

            # left half sorted
            if nums[lo] <= nums[mid]:
                if nums[lo] <= target < nums[mid]:
                    hi = mid - 1
                else:
                    lo = mid + 1
            else:  # right half sorted
                if nums[mid] < target <= nums[hi]:
                    lo = mid + 1
                else:
                    hi = mid - 1
        return -1


def run_tests() -> None:
    sol = Solution()
    assert sol.search([4, 5, 6, 7, 0, 1, 2], 0) == 4
    assert sol.search([4, 5, 6, 7, 0, 1, 2], 3) == -1
    assert sol.search([1], 0) == -1
    assert sol.search([], 1) == -1


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


