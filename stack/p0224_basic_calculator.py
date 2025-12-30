"""224. Basic Calculator

Link: https://leetcode.com/problems/basic-calculator/

Problem:
Evaluate a simple expression string containing non-negative integers, '+', '-', '(',
')', and spaces.

Approach (stack of signs):
We scan left-to-right building numbers.
Maintain:
- `res`: current accumulated result
- `sign`: +1 or -1 applied to the next number
- stack holds previous (res, sign) when entering parentheses

When we see '(':
  push (res, sign), reset res=0, sign=+1
When we see ')':
  prev_res, prev_sign = pop()
  res = prev_res + prev_sign * res

Complexity:
- Time: O(n)
- Space: O(n) (nesting depth)
"""

from __future__ import annotations

import sys


class Solution:
    def calculate(self, s: str) -> int:
        res = 0
        sign = 1
        num = 0
        stack: list[tuple[int, int]] = []

        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(ch)
            elif ch in "+-":
                res += sign * num
                num = 0
                sign = 1 if ch == "+" else -1
            elif ch == "(":
                # Save state before parentheses.
                stack.append((res, sign))
                res = 0
                sign = 1
                num = 0
            elif ch == ")":
                res += sign * num
                num = 0
                prev_res, prev_sign = stack.pop()
                res = prev_res + prev_sign * res
            else:
                # space
                continue

        res += sign * num
        return res


def run_tests() -> None:
    sol = Solution()
    assert sol.calculate("1 + 1") == 2
    assert sol.calculate(" 2-1 + 2 ") == 3
    assert sol.calculate("(1+(4+5+2)-3)+(6+8)") == 23
    assert sol.calculate("0") == 0
    assert sol.calculate("2147483647") == 2147483647


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
