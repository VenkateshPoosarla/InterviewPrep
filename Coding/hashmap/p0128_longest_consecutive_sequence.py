"""128. Longest Consecutive Sequence

Link: https://leetcode.com/problems/longest-consecutive-sequence/

Problem:
Given an unsorted array of integers, return the length of the longest consecutive
elements sequence.

Approach (hash set + start-of-sequence):
Put all numbers in a set.
Only start counting from numbers that are the beginning of a streak (x-1 not in set).
Then count x, x+1, x+2, ... while present.

Each number is visited O(1) times overall because we only extend sequences from starts.

Complexity:
- Time: O(n) average
- Space: O(n)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        s = set(nums)
        best = 0
        for x in s:
            if x - 1 in s:
                continue  # not a start
            y = x
            while y in s:
                y += 1
            best = max(best, y - x)
        return best


def run_tests() -> None:
    sol = Solution()
    assert sol.longestConsecutive([100, 4, 200, 1, 3, 2]) == 4
    assert sol.longestConsecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9
    assert sol.longestConsecutive([]) == 0
    assert sol.longestConsecutive([1, 2, 0, 1]) == 3


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
