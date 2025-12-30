"""172. Factorial Trailing Zeroes

Link: https://leetcode.com/problems/factorial-trailing-zeroes/

Problem:
Return the number of trailing zeros in n!.

Key idea:
Trailing zeros come from factors of 10 = 2 * 5.
There are always more 2s than 5s in n!, so we count how many 5s appear:
  floor(n/5) + floor(n/25) + floor(n/125) + ...

Complexity:
- Time: O(log_5(n))
- Space: O(1)
"""

from __future__ import annotations

import sys


class Solution:
    def trailingZeroes(self, n: int) -> int:
        count = 0
        while n:
            n //= 5
            count += n
        return count


def run_tests() -> None:
    sol = Solution()
    assert sol.trailingZeroes(3) == 0
    assert sol.trailingZeroes(5) == 1
    assert sol.trailingZeroes(10) == 2
    assert sol.trailingZeroes(25) == 6


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
