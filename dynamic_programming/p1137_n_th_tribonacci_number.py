"""1137. N-th Tribonacci Number

Link: https://leetcode.com/problems/n-th-tribonacci-number/

DP recurrence:
T0=0, T1=1, T2=1
Tn = T(n-1) + T(n-2) + T(n-3)
Keep rolling window of 3 values.
"""

from __future__ import annotations


class Solution:
    def tribonacci(self, n: int) -> int:
        if n == 0:
            return 0
        if n == 1 or n == 2:
            return 1
        a, b, c = 0, 1, 1
        for _ in range(3, n + 1):
            a, b, c = b, c, a + b + c
        return c


def run_tests() -> None:
    sol = Solution()
    assert sol.tribonacci(0) == 0
    assert sol.tribonacci(1) == 1
    assert sol.tribonacci(2) == 1
    assert sol.tribonacci(4) == 4
    assert sol.tribonacci(25) == 1389537


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


