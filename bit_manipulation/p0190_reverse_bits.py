"""190. Reverse Bits

Link: https://leetcode.com/problems/reverse-bits/

Problem:
Reverse bits of a given 32-bit unsigned integer.

Approach:
Build result bit-by-bit:
- Repeat 32 times:
  - take lowest bit of n
  - shift result left, append that bit
  - shift n right

Complexity:
- Time: O(32)
- Space: O(1)
"""

from __future__ import annotations

import sys


class Solution:
    def reverseBits(self, n: int) -> int:
        res = 0
        for _ in range(32):
            res = (res << 1) | (n & 1)
            n >>= 1
        return res


def run_tests() -> None:
    sol = Solution()
    assert sol.reverseBits(int("00000010100101000001111010011100", 2)) == int("00111001011110000010100101000000", 2)
    assert sol.reverseBits(int("11111111111111111111111111111101", 2)) == int("10111111111111111111111111111111", 2)


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
