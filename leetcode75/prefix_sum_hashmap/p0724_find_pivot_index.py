"""724. Find Pivot Index

Link: https://leetcode.com/problems/find-pivot-index/

Pivot index i satisfies:
sum(nums[0:i]) == sum(nums[i+1:])

We can do this in one pass using total sum and a running left sum.
Right sum = total - left - nums[i]
"""

from __future__ import annotations

from typing import List


class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        total = sum(nums)
        left = 0
        for i, x in enumerate(nums):
            right = total - left - x
            if left == right:
                return i
            left += x
        return -1


def run_tests() -> None:
    sol = Solution()

    assert sol.pivotIndex([1, 7, 3, 6, 5, 6]) == 3
    assert sol.pivotIndex([1, 2, 3]) == -1
    assert sol.pivotIndex([2, 1, -1]) == 0
    assert sol.pivotIndex([]) == -1


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


