"""162. Find Peak Element

Link: https://leetcode.com/problems/find-peak-element/

Problem:
A peak element is an element strictly greater than its neighbors.
Given an array nums where nums[i] != nums[i+1], return an index of any peak element.

Approach (binary search on slope):
Compare mid with mid+1:
- If nums[mid] < nums[mid+1], we're on an ascending slope, so a peak exists on the right.
- Else, a peak exists on the left (including mid).

Complexity:
- Time: O(log n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        lo, hi = 0, len(nums) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] < nums[mid + 1]:
                lo = mid + 1
            else:
                hi = mid
        return lo


def run_tests() -> None:
    sol = Solution()
    idx = sol.findPeakElement([1, 2, 3, 1])
    assert idx == 2
    idx = sol.findPeakElement([1, 2, 1, 3, 5, 6, 4])
    assert idx in {1, 5}
    assert sol.findPeakElement([1]) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
