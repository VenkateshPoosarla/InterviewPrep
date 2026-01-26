"""191. Number of 1 Bits

Link: https://leetcode.com/problems/number-of-1-bits/

Problem:
Return the number of '1' bits (Hamming weight) in a 32-bit unsigned integer.

Approach (Brian Kernighan trick):
Repeatedly clear the lowest set bit:
  n &= n - 1
Each operation removes one '1' bit.

Complexity:
- Time: O(#set bits) <= 32
- Space: O(1)
"""

from __future__ import annotations

import sys


class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            n &= n - 1
            count += 1
        return count


def run_tests() -> None:
    sol = Solution()
    assert sol.hammingWeight(int("00000000000000000000000000001011", 2)) == 3
    assert sol.hammingWeight(int("00000000000000000000000010000000", 2)) == 1
    assert sol.hammingWeight(int("11111111111111111111111111111101", 2)) == 31


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
