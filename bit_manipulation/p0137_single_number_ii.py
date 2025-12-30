"""137. Single Number II

Link: https://leetcode.com/problems/single-number-ii/

Problem:
Every element appears three times except for one, which appears exactly once.
Find the single one.

Approach (bit counting mod 3):
For each bit position 0..31, count how many numbers have that bit set.
If count % 3 == 1, that bit is set in the unique number.
Handle negative numbers via 32-bit signed conversion.

Complexity:
- Time: O(32*n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for bit in range(32):
            cnt = 0
            mask = 1 << bit
            for x in nums:
                if x & mask:
                    cnt += 1
            if cnt % 3:
                res |= mask
        # convert to signed 32-bit
        if res >= 1 << 31:
            res -= 1 << 32
        return res


def run_tests() -> None:
    sol = Solution()
    assert sol.singleNumber([2, 2, 3, 2]) == 3
    assert sol.singleNumber([0, 1, 0, 1, 0, 1, 99]) == 99
    assert sol.singleNumber([-2, -2, -2, -7]) == -7


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
