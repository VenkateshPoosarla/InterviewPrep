"""202. Happy Number

Link: https://leetcode.com/problems/happy-number/

Problem:
Starting with n, repeatedly replace n by the sum of the squares of its digits.
Return True if this process ends in 1; otherwise it loops forever.

Approach (cycle detection with a set):
Compute next(n). If we ever see a number twice, we're in a loop -> not happy.

Complexity:
- Time: small (numbers quickly fall below a bounded range)
- Space: O(#seen)
"""

from __future__ import annotations

import sys


class Solution:
    def isHappy(self, n: int) -> bool:
        def nxt(x: int) -> int:
            s = 0
            while x:
                x, d = divmod(x, 10)
                s += d * d
            return s

        seen: set[int] = set()
        while n != 1 and n not in seen:
            seen.add(n)
            n = nxt(n)
        return n == 1


def run_tests() -> None:
    sol = Solution()
    assert sol.isHappy(19) is True
    assert sol.isHappy(2) is False
    assert sol.isHappy(1) is True


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
