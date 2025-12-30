"""201. Bitwise AND of Numbers Range

Link: https://leetcode.com/problems/bitwise-and-of-numbers-range/

Problem:
Given two integers left and right, return the bitwise AND of all numbers in [left, right].

Key idea:
Any bit that changes within the range becomes 0 in the AND result.
So we need the common prefix (highest bits that stay the same) of left and right.

Approach (shift to common prefix):
Right-shift both left and right until they are equal, counting shifts.
Then shift back.

Complexity:
- Time: O(32)
- Space: O(1)
"""

from __future__ import annotations

import sys


class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        shift = 0
        while left != right:
            left >>= 1
            right >>= 1
            shift += 1
        return left << shift


def run_tests() -> None:
    sol = Solution()
    assert sol.rangeBitwiseAnd(5, 7) == 4
    assert sol.rangeBitwiseAnd(0, 0) == 0
    assert sol.rangeBitwiseAnd(1, 2147483647) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
