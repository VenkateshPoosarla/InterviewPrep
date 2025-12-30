"""80. Remove Duplicates from Sorted Array II

Link: https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/

Problem:
Given a sorted array, remove duplicates in-place such that each element appears at most
twice. Return the new length `k`.

Approach:
Maintain a `write` pointer for the next valid slot.
We can always keep the first two elements; for the rest, we only write nums[i] if it
differs from nums[write-2] (meaning we haven't already kept two of that value).

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        write = 0
        for x in nums:
            if write < 2 or x != nums[write - 2]:
                nums[write] = x
                write += 1
        return write


def run_tests() -> None:
    sol = Solution()

    nums = [1, 1, 1, 2, 2, 3]
    k = sol.removeDuplicates(nums)
    assert k == 5
    assert nums[:k] == [1, 1, 2, 2, 3]

    nums = [0, 0, 1, 1, 1, 1, 2, 3, 3]
    k = sol.removeDuplicates(nums)
    assert k == 7
    assert nums[:k] == [0, 0, 1, 1, 2, 3, 3]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
