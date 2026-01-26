"""56. Merge Intervals

Link: https://leetcode.com/problems/merge-intervals/

Problem:
Given an array of intervals where intervals[i] = [start, end], merge all overlapping
intervals and return an array of the non-overlapping intervals that cover all input intervals.

Approach:
Sort by start. Keep a `merged` list:
- If current interval starts after the last merged ends, append it.
- Otherwise, extend the last merged's end = max(end, current_end).

Complexity:
- Time: O(n log n) due to sorting
- Space: O(n) for output
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []
        intervals.sort(key=lambda x: x[0])
        merged: List[List[int]] = [intervals[0][:]]

        for s, e in intervals[1:]:
            last = merged[-1]
            if s > last[1]:
                merged.append([s, e])
            else:
                last[1] = max(last[1], e)
        return merged


def run_tests() -> None:
    sol = Solution()
    assert sol.merge([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert sol.merge([[1, 4], [4, 5]]) == [[1, 5]]
    assert sol.merge([]) == []


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
