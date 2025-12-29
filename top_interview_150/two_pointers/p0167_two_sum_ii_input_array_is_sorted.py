"""167. Two Sum II - Input Array Is Sorted

Link: https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/

Problem:
Given a 1-indexed sorted array `numbers`, find two numbers such that they add up to `target`.
Return their indices [index1, index2] (1-indexed), with index1 < index2.
Exactly one solution exists.

Approach (two pointers):
Because the array is sorted:
- If numbers[l] + numbers[r] is too small, increase l.
- If too large, decrease r.

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        l, r = 0, len(numbers) - 1
        while l < r:
            s = numbers[l] + numbers[r]
            if s == target:
                return [l + 1, r + 1]
            if s < target:
                l += 1
            else:
                r -= 1
        raise ValueError("No solution (problem guarantees one).")


def run_tests() -> None:
    sol = Solution()
    assert sol.twoSum([2, 7, 11, 15], 9) == [1, 2]
    assert sol.twoSum([2, 3, 4], 6) == [1, 3]
    assert sol.twoSum([-1, 0], -1) == [1, 2]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
