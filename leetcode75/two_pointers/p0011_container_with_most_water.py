"""11. Container With Most Water

Link: https://leetcode.com/problems/container-with-most-water/

Two pointers:
Start with widest container (l=0, r=n-1).
Area = min(h[l], h[r]) * (r-l).
To possibly improve, move the pointer with the smaller height inward (only that can increase min-height).

Visual:
  l .... r
  width shrinks each step, so we need min-height to increase to compensate.
"""

from __future__ import annotations

from typing import List


class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        best = 0
        while l < r:
            h = min(height[l], height[r])
            best = max(best, h * (r - l))
            if height[l] <= height[r]:
                l += 1
            else:
                r -= 1
        return best


def run_tests() -> None:
    sol = Solution()

    assert sol.maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49
    assert sol.maxArea([1, 1]) == 1
    assert sol.maxArea([4, 3, 2, 1, 4]) == 16
    assert sol.maxArea([1, 2, 1]) == 2


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


