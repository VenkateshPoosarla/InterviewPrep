"""66. Plus One

Link: https://leetcode.com/problems/plus-one/

Problem:
Given a non-empty array of digits representing a non-negative integer (most-significant digit first),
increment the integer by one and return the resulting digits.

Approach:
Add carry=1 from the end:
- digits[i] = (digits[i] + carry) % 10
- carry becomes 1 only if we rolled over from 9 -> 0
If carry remains, prepend 1.

Complexity:
- Time: O(n)
- Space: O(1) extra (output list may grow by 1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        carry = 1
        for i in range(len(digits) - 1, -1, -1):
            s = digits[i] + carry
            digits[i] = s % 10
            carry = s // 10
            if carry == 0:
                break
        if carry:
            return [1] + digits
        return digits


def run_tests() -> None:
    sol = Solution()
    assert sol.plusOne([1, 2, 3]) == [1, 2, 4]
    assert sol.plusOne([4, 3, 2, 1]) == [4, 3, 2, 2]
    assert sol.plusOne([9]) == [1, 0]
    assert sol.plusOne([9, 9, 9]) == [1, 0, 0, 0]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
