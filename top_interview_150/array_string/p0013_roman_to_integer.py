"""13. Roman to Integer

Link: https://leetcode.com/problems/roman-to-integer/

Problem:
Convert a Roman numeral string into an integer.

Rules:
Most symbols add, but a smaller value before a larger value means subtraction:
  IV = 4, IX = 9, XL = 40, XC = 90, CD = 400, CM = 900

Approach:
Scan left to right:
- If s[i] is smaller than s[i+1], subtract it
- Else add it

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys


class Solution:
    def romanToInt(self, s: str) -> int:
        val = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000,
        }

        total = 0
        for i, ch in enumerate(s):
            cur = val[ch]
            nxt = val[s[i + 1]] if i + 1 < len(s) else 0
            if cur < nxt:
                total -= cur
            else:
                total += cur
        return total


def run_tests() -> None:
    sol = Solution()
    assert sol.romanToInt("III") == 3
    assert sol.romanToInt("IV") == 4
    assert sol.romanToInt("IX") == 9
    assert sol.romanToInt("LVIII") == 58
    assert sol.romanToInt("MCMXCIV") == 1994


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
