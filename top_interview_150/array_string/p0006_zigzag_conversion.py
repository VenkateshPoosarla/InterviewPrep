"""6. Zigzag Conversion

Link: https://leetcode.com/problems/zigzag-conversion/

Problem:
Write the characters of `s` in a zigzag pattern on `numRows` rows and then read
row-by-row.

Example (numRows=3):
  PAYPALISHIRING
  P   A   H   N
  A P L S I I G
  Y   I   R
  => "PAHNAPLSIIGYIR"

Approach (simulate rows):
Keep a current row index and a direction (+1 down, -1 up). Append each character to
its row, flipping direction at row 0 and row numRows-1.

Complexity:
- Time: O(n)
- Space: O(n)
"""

from __future__ import annotations

import sys


class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows <= 1 or numRows >= len(s):
            return s

        rows = [[] for _ in range(numRows)]
        row = 0
        direction = 1
        for ch in s:
            rows[row].append(ch)
            if row == 0:
                direction = 1
            elif row == numRows - 1:
                direction = -1
            row += direction

        return "".join("".join(r) for r in rows)


def run_tests() -> None:
    sol = Solution()
    assert sol.convert("PAYPALISHIRING", 3) == "PAHNAPLSIIGYIR"
    assert sol.convert("PAYPALISHIRING", 4) == "PINALSIGYAHRPI"
    assert sol.convert("A", 1) == "A"
    assert sol.convert("AB", 10) == "AB"


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
