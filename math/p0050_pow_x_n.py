"""50. Pow(x, n)

Link: https://leetcode.com/problems/powx-n/

Problem:
Implement pow(x, n), which calculates x raised to the power n (i.e., \(x^n\)).

Approach (fast exponentiation):
Use binary exponentiation:
- If n < 0, compute (1/x)^{-n}
- Repeatedly square the base and multiply into result when the current bit of n is 1.

Complexity:
- Time: O(log |n|)
- Space: O(1)
"""

from __future__ import annotations

import sys


class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1.0
        if n < 0:
            x = 1.0 / x
            n = -n

        res = 1.0
        base = x
        while n:
            if n & 1:
                res *= base
            base *= base
            n >>= 1
        return res


def run_tests() -> None:
    sol = Solution()
    assert abs(sol.myPow(2.0, 10) - 1024.0) < 1e-9
    assert abs(sol.myPow(2.1, 3) - 9.261) < 1e-9
    assert abs(sol.myPow(2.0, -2) - 0.25) < 1e-9


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
