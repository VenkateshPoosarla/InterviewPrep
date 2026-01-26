"""12. Integer to Roman

Link: https://leetcode.com/problems/integer-to-roman/

Problem:
Convert an integer to a Roman numeral (1 <= num <= 3999).

Approach (greedy):
Use a descending list of (value, symbol) including the subtractive forms (900=CM, etc).
Repeatedly take as many of the largest value as possible.

Complexity:
- Time: O(1) (bounded by constant roman symbol list)
- Space: O(1)
"""

from __future__ import annotations

import sys


class Solution:
    def intToRoman(self, num: int) -> str:
        pairs = [
            (1000, "M"),
            (900, "CM"),
            (500, "D"),
            (400, "CD"),
            (100, "C"),
            (90, "XC"),
            (50, "L"),
            (40, "XL"),
            (10, "X"),
            (9, "IX"),
            (5, "V"),
            (4, "IV"),
            (1, "I"),
        ]

        out: list[str] = []
        for value, sym in pairs:
            if num == 0:
                break
            count, num = divmod(num, value)
            if count:
                out.append(sym * count)
        return "".join(out)


def run_tests() -> None:
    sol = Solution()
    # assert sol.intToRoman(3) == "III"
    # assert sol.intToRoman(4) == "IV"
    # assert sol.intToRoman(9) == "IX"
    assert sol.intToRoman(58) == "LVIII"
    assert sol.intToRoman(1994) == "MCMXCIV"


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
