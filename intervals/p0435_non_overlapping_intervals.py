"""435. Non-overlapping Intervals

Link: https://leetcode.com/problems/non-overlapping-intervals/

Goal: remove the minimum number of intervals to make the rest non-overlapping.

Greedy:
Sort intervals by end time.
Always keep the interval with the earliest finishing time (maximizes room for future intervals).
Count how many intervals we can keep; removals = n - kept.
"""

from __future__ import annotations

from typing import List


class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        intervals.sort(key=lambda x: x[1])
        kept = 1
        end = intervals[0][1]
        for s, e in intervals[1:]:
            if s >= end:
                kept += 1
                end = e
        return len(intervals) - kept


def run_tests() -> None:
    sol = Solution()

    assert sol.eraseOverlapIntervals([[1, 2], [2, 3], [3, 4], [1, 3]]) == 1
    assert sol.eraseOverlapIntervals([[1, 2], [1, 2], [1, 2]]) == 2
    assert sol.eraseOverlapIntervals([[1, 2], [2, 3]]) == 0
    assert sol.eraseOverlapIntervals([]) == 0


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


