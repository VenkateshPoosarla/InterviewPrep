"""57. Insert Interval

Link: https://leetcode.com/problems/insert-interval/

Problem:
Given a list of non-overlapping intervals sorted by start time, insert a new interval
and merge if necessary.

Approach:
1) Add all intervals that end before newInterval starts.
2) Merge all overlapping intervals with newInterval.
3) Add the remaining intervals.

Complexity:
- Time: O(n)
- Space: O(n) for output
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        res: List[List[int]] = []
        i = 0
        n = len(intervals)

        # 1) non-overlapping before
        while i < n and intervals[i][1] < newInterval[0]:
            res.append(intervals[i])
            i += 1

        # 2) merge overlaps
        start, end = newInterval
        while i < n and intervals[i][0] <= end:
            start = min(start, intervals[i][0])
            end = max(end, intervals[i][1])
            i += 1
        res.append([start, end])

        # 3) rest
        while i < n:
            res.append(intervals[i])
            i += 1

        return res


def run_tests() -> None:
    sol = Solution()
    assert sol.insert([[1, 3], [6, 9]], [2, 5]) == [[1, 5], [6, 9]]
    assert sol.insert([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8]) == [[1, 2], [3, 10], [12, 16]]
    assert sol.insert([], [5, 7]) == [[5, 7]]
    assert sol.insert([[1, 5]], [2, 3]) == [[1, 5]]
    assert sol.insert([[1, 5]], [6, 8]) == [[1, 5], [6, 8]]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
