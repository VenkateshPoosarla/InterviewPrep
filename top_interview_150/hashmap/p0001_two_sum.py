"""1. Two Sum

Link: https://leetcode.com/problems/two-sum/

Problem:
Given an array `nums` and an integer `target`, return indices of the two numbers such
that they add up to target. Exactly one solution exists; you may not use the same
element twice.

Approach (hash map):
As we scan nums, store seen value -> index.
For current x at i, we need y = target - x. If y is already seen, answer is [seen[y], i].

Complexity:
- Time: O(n)
- Space: O(n)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen: dict[int, int] = {}
        for i, x in enumerate(nums):
            y = target - x
            if y in seen:
                return [seen[y], i]
            seen[x] = i
        raise ValueError("No solution (problem guarantees one).")


def run_tests() -> None:
    sol = Solution()
    assert sol.twoSum([2, 7, 11, 15], 9) == [0, 1]
    assert sol.twoSum([3, 2, 4], 6) == [1, 2]
    assert sol.twoSum([3, 3], 6) == [0, 1]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
