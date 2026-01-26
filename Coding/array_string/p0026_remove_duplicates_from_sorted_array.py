"""26. Remove Duplicates from Sorted Array

Link: https://leetcode.com/problems/remove-duplicates-from-sorted-array/

Problem:
Given a sorted array, remove duplicates in-place such that each unique element
appears only once. Return the number of unique elements `k`.

Approach (two pointers):
- `write` marks the position to write the next new unique value.
- Scan with `read`; when nums[read] differs from the last kept value, write it.

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0

        write = 1  # first element is always unique
        for read in range(1, len(nums)):
            if nums[read] != nums[write - 1]:
                nums[write] = nums[read]
                write += 1
        return write


def run_tests() -> None:
    sol = Solution()

    nums = [1, 1, 2]
    k = sol.removeDuplicates(nums)
    assert k == 2
    assert nums[:k] == [1, 2]

    nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    k = sol.removeDuplicates(nums)
    assert k == 5
    assert nums[:k] == [0, 1, 2, 3, 4]

    nums = []
    k = sol.removeDuplicates(nums)
    assert k == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
