"""1431. Kids With the Greatest Number of Candies

Link: https://leetcode.com/problems/kids-with-the-greatest-number-of-candies/

Idea:
Let M = max(candies). For each kid i, check candies[i] + extraCandies >= M.
"""

from __future__ import annotations

from typing import List


class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        m = max(candies) if candies else 0
        return [c + extraCandies >= m for c in candies]


def run_tests() -> None:
    s = Solution()

    assert s.kidsWithCandies([2, 3, 5, 1, 3], 3) == [True, True, True, False, True]
    assert s.kidsWithCandies([4, 2, 1, 1, 2], 1) == [True, False, False, False, False]
    assert s.kidsWithCandies([12, 1, 12], 10) == [True, False, True]
    assert s.kidsWithCandies([], 5) == []


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


