"""452. Minimum Number of Arrows to Burst Balloons

Link: https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/

Each balloon is an interval [xstart, xend]. One arrow shot at position x bursts all
balloons where xstart <= x <= xend.

Greedy:
Sort by end coordinate; shoot an arrow at current end, and reuse it for all balloons
whose start <= arrow_pos. Otherwise, need a new arrow.
"""

from __future__ import annotations

from typing import List


class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if not points:
            return 0
        points.sort(key=lambda p: p[1])
        arrows = 1
        arrow_pos = points[0][1]
        for s, e in points[1:]:
            if s > arrow_pos:
                arrows += 1
                arrow_pos = e
        return arrows


def run_tests() -> None:
    sol = Solution()

    assert sol.findMinArrowShots([[10, 16], [2, 8], [1, 6], [7, 12]]) == 2
    assert sol.findMinArrowShots([[1, 2], [3, 4], [5, 6], [7, 8]]) == 4
    assert sol.findMinArrowShots([[1, 2], [2, 3], [3, 4], [4, 5]]) == 2
    assert sol.findMinArrowShots([]) == 0


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


