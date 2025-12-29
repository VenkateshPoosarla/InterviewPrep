"""1318. Minimum Flips to Make a OR b Equal to c

Link: https://leetcode.com/problems/minimum-flips-to-make-a-or-b-equal-to-c/

Bit-by-bit:
For each bit position:
- If c has 0, then both a and b must be 0:
    flips += (a_bit == 1) + (b_bit == 1)
- If c has 1, then at least one of a or b must be 1:
    if a_bit==0 and b_bit==0: flips += 1
"""

from __future__ import annotations


class Solution:
    def minFlips(self, a: int, b: int, c: int) -> int:
        flips = 0
        for _ in range(32):  # constraints fit in 32 bits
            abit = a & 1
            bbit = b & 1
            cbit = c & 1

            if cbit == 0:
                flips += abit + bbit
            else:
                if abit == 0 and bbit == 0:
                    flips += 1

            a >>= 1
            b >>= 1
            c >>= 1

        return flips


def run_tests() -> None:
    sol = Solution()
    assert sol.minFlips(2, 6, 5) == 3
    assert sol.minFlips(4, 2, 7) == 1
    assert sol.minFlips(1, 2, 3) == 0


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


