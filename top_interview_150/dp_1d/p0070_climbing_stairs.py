"""70. Climbing Stairs

Link: https://leetcode.com/problems/climbing-stairs/

Problem:
You can climb 1 or 2 steps. How many distinct ways to reach the n-th step?

DP recurrence:
ways[n] = ways[n-1] + ways[n-2]

Approach:
Keep only the last two values (Fibonacci-style).

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys


class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        a, b = 1, 2
        for _ in range(3, n + 1):
            a, b = b, a + b
        return b


def run_tests() -> None:
    sol = Solution()
    assert sol.climbStairs(1) == 1
    assert sol.climbStairs(2) == 2
    assert sol.climbStairs(3) == 3
    assert sol.climbStairs(4) == 5


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
