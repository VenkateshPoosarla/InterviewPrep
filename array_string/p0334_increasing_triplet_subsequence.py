"""334. Increasing Triplet Subsequence

Link: https://leetcode.com/problems/increasing-triplet-subsequence/

Greedy:
Maintain the smallest possible first and second elements of an increasing pair.

- first = smallest seen so far
- second = smallest possible value > first
If we ever find x > second, we have first < second < x => triplet exists.

Visual:
  nums: 2, 1, 5, 0, 4, 6
  first: 2 -> 1 -> 0
  second: inf -> 5 -> 4
  then 6 > 4 => True
"""

from __future__ import annotations

from typing import List


class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        first = float("inf")
        second = float("inf")

        for x in nums:
            if x <= first:
                first = x
            elif x <= second:
                second = x
            else:
                return True

        return False


def run_tests() -> None:
    sol = Solution()

    assert sol.increasingTriplet([1, 2, 3, 4, 5]) is True
    assert sol.increasingTriplet([5, 4, 3, 2, 1]) is False
    assert sol.increasingTriplet([2, 1, 5, 0, 4, 6]) is True
    assert sol.increasingTriplet([2, 4, -2, -3]) is False
    assert sol.increasingTriplet([1, 1, 1, 1]) is False
    assert sol.increasingTriplet([1, 2]) is False


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


