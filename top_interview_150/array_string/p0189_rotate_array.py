"""189. Rotate Array

Link: https://leetcode.com/problems/rotate-array/

Problem:
Rotate the array to the right by k steps, in-place.

Approach (3 reversals):
For k % n:
- Reverse the whole array
- Reverse the first k elements
- Reverse the remaining n-k elements

Example:
  [1,2,3,4,5,6,7], k=3
  reverse all -> [7,6,5,4,3,2,1]
  reverse first 3 -> [5,6,7,4,3,2,1]
  reverse rest -> [5,6,7,1,2,3,4]

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        n = len(nums)
        if n == 0:
            return
        k %= n
        if k == 0:
            return

        def reverse(lo: int, hi: int) -> None:
            while lo < hi:
                nums[lo], nums[hi] = nums[hi], nums[lo]
                lo += 1
                hi -= 1

        reverse(0, n - 1)
        reverse(0, k - 1)
        reverse(k, n - 1)


def run_tests() -> None:
    sol = Solution()

    nums = [1, 2, 3, 4, 5, 6, 7]
    sol.rotate(nums, 3)
    assert nums == [5, 6, 7, 1, 2, 3, 4]

    nums = [-1, -100, 3, 99]
    sol.rotate(nums, 2)
    assert nums == [3, 99, -1, -100]

    nums = [1, 2]
    sol.rotate(nums, 2)
    assert nums == [1, 2]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
