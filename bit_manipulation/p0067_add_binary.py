"""67. Add Binary

Link: https://leetcode.com/problems/add-binary/

Problem:
Given two binary strings a and b, return their sum as a binary string.

Approach (manual addition with carry):
Walk from the end of both strings, add digits + carry, build result backwards.

Complexity:
- Time: O(n + m)
- Space: O(n + m)
"""

from __future__ import annotations

import sys


class Solution:
    def addBinary(self, a: str, b: str) -> str:
        i, j = len(a) - 1, len(b) - 1
        carry = 0
        out: list[str] = []

        while i >= 0 or j >= 0 or carry:
            s = carry
            if i >= 0:
                s += ord(a[i]) - ord("0")
                i -= 1
            if j >= 0:
                s += ord(b[j]) - ord("0")
                j -= 1
            carry, bit = divmod(s, 2)
            out.append(str(bit))

        out.reverse()
        return "".join(out)


def run_tests() -> None:
    sol = Solution()
    assert sol.addBinary("11", "1") == "100"
    assert sol.addBinary("1010", "1011") == "10101"
    assert sol.addBinary("0", "0") == "0"


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
