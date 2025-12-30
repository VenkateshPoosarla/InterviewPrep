"""704. Binary Search

Link: https://leetcode.com/problems/binary-search/

Classic binary search on a sorted array.
Return index of target or -1.
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
            if nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1


def run_tests() -> None:
    sol = Solution()
    assert sol.search([-1, 0, 3, 5, 9, 12], 9) == 4
    assert sol.search([-1, 0, 3, 5, 9, 12], 2) == -1
    assert sol.search([], 1) == -1
    assert sol.search([1], 1) == 0


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


