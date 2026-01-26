"""35. Search Insert Position

Link: https://leetcode.com/problems/search-insert-position/

Problem:
Given a sorted array of distinct integers and a target, return the index if found.
If not, return the index where it would be inserted to keep order.

Approach (binary search / lower_bound):
Find the first position where nums[pos] >= target.

Complexity:
- Time: O(log n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        lo, hi = 0, len(nums)
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo


def run_tests() -> None:
    sol = Solution()
    assert sol.searchInsert([1, 3, 5, 6], 5) == 2
    assert sol.searchInsert([1, 3, 5, 6], 2) == 1
    assert sol.searchInsert([1, 3, 5, 6], 7) == 4
    assert sol.searchInsert([1, 3, 5, 6], 0) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
