"""4. Median of Two Sorted Arrays

Link: https://leetcode.com/problems/median-of-two-sorted-arrays/

Problem:
Given two sorted arrays nums1 and nums2, return the median of the two sorted arrays.
Overall runtime must be O(log(m+n)).

Approach (binary search partition):
Binary search on the smaller array to find a partition i (in nums1) and j (in nums2)
such that:
  left side has (m+n+1)//2 elements
  max(left) <= min(right)

Then:
- If total length odd: median = max(left)
- Else: median = (max(left) + min(right)) / 2

Complexity:
- Time: O(log(min(m,n)))
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # Ensure nums1 is the smaller array.
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1

        m, n = len(nums1), len(nums2)
        total_left = (m + n + 1) // 2

        lo, hi = 0, m
        while lo <= hi:
            i = (lo + hi) // 2
            j = total_left - i

            left1 = nums1[i - 1] if i > 0 else float("-inf")
            right1 = nums1[i] if i < m else float("inf")
            left2 = nums2[j - 1] if j > 0 else float("-inf")
            right2 = nums2[j] if j < n else float("inf")

            if left1 <= right2 and left2 <= right1:
                if (m + n) % 2 == 1:
                    return float(max(left1, left2))
                return (max(left1, left2) + min(right1, right2)) / 2.0
            if left1 > right2:
                hi = i - 1
            else:
                lo = i + 1

        raise RuntimeError("Unreachable")


def run_tests() -> None:
    sol = Solution()
    assert sol.findMedianSortedArrays([1, 3], [2]) == 2.0
    assert sol.findMedianSortedArrays([1, 2], [3, 4]) == 2.5
    assert sol.findMedianSortedArrays([], [1]) == 1.0
    assert sol.findMedianSortedArrays([2], []) == 2.0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
