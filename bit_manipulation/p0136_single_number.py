"""136. Single Number

Link: https://leetcode.com/problems/single-number/

Problem:
Every element appears twice except for one. Find that single one.

Approach (XOR):
XOR properties:
- a ^ a = 0
- a ^ 0 = a
- XOR is commutative/associative
So XORing all elements cancels pairs and leaves the unique element.

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        x = 0
        for v in nums:
            x ^= v
        return x


def run_tests() -> None:
    sol = Solution()
    assert sol.singleNumber([2, 2, 1]) == 1
    assert sol.singleNumber([4, 1, 2, 1, 2]) == 4
    assert sol.singleNumber([1]) == 1


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
