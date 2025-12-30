"""88. Merge Sorted Array

Link: https://leetcode.com/problems/merge-sorted-array/

Problem:
You are given two sorted integer arrays `nums1` and `nums2`, where `nums1` has enough
trailing space to hold all elements of `nums2`. Merge `nums2` into `nums1` in-place.

Approach (3 pointers from the end):
- Let i point at the last real element in nums1 (m-1)
- Let j point at the last element in nums2 (n-1)
- Let k point at the last slot in nums1 (m+n-1)
- Fill nums1[k] with the larger of nums1[i] and nums2[j], moving pointers accordingly.

Why from the end?
Merging from the front would overwrite elements in nums1 that we still need to compare.

Complexity:
- Time: O(m + n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        i = m - 1
        j = n - 1
        k = m + n - 1

        while j >= 0:
            # If nums1 is exhausted, we must take from nums2.
            if i < 0 or nums2[j] >= nums1[i]:
                nums1[k] = nums2[j]
                j -= 1
            else:
                nums1[k] = nums1[i]
                i -= 1
            k -= 1


def run_tests() -> None:
    sol = Solution()

    nums1 = [1, 2, 3, 0, 0, 0]
    sol.merge(nums1, 3, [2, 5, 6], 3)
    assert nums1 == [1, 2, 2, 3, 5, 6]

    nums1 = [1]
    sol.merge(nums1, 1, [], 0)
    assert nums1 == [1]

    nums1 = [0]
    sol.merge(nums1, 0, [1], 1)
    assert nums1 == [1]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
