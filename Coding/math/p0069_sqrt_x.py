"""69. Sqrt(x)

Link: https://leetcode.com/problems/sqrtx/

Problem:
Given a non-negative integer x, compute and return floor(sqrt(x)).

Approach (binary search):
Search integer m such that m^2 <= x < (m+1)^2.

Complexity:
- Time: O(log x)
- Space: O(1)
"""

from __future__ import annotations

import sys


class Solution:
    def mySqrt(self, x: int) -> int:
        if x < 2:
            return x
        lo, hi = 1, x // 2
        while lo <= hi:
            mid = (lo + hi) // 2
            sq = mid * mid
            if sq == x:
                return mid
            if sq < x:
                lo = mid + 1
            else:
                hi = mid - 1
        return hi


def run_tests() -> None:
    sol = Solution()
    assert sol.mySqrt(0) == 0
    assert sol.mySqrt(1) == 1
    assert sol.mySqrt(4) == 2
    assert sol.mySqrt(8) == 2
    assert sol.mySqrt(2147395599) == 46339


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
