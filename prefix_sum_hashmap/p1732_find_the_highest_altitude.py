"""1732. Find the Highest Altitude

Link: https://leetcode.com/problems/find-the-highest-altitude/

We start at altitude 0. gain[i] adds to current altitude.
Track running sum and maximum.
"""

from __future__ import annotations

from typing import List


class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        cur = 0
        best = 0
        for g in gain:
            cur += g
            if cur > best:
                best = cur
        return best


def run_tests() -> None:
    sol = Solution()

    assert sol.largestAltitude([-5, 1, 5, 0, -7]) == 1
    assert sol.largestAltitude([-4, -3, -2, -1, 4, 3, 2]) == 0
    assert sol.largestAltitude([]) == 0


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


