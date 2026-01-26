"""150. Evaluate Reverse Polish Notation

Link: https://leetcode.com/problems/evaluate-reverse-polish-notation/

Problem:
Evaluate the value of an arithmetic expression in Reverse Polish Notation (postfix).
Supported operators: +, -, *, / (integer division truncates toward zero).

Approach (stack):
For each token:
- If it's a number: push
- Else it's an operator: pop b, pop a, compute a op b, push result

Complexity:
- Time: O(n)
- Space: O(n)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack: list[int] = []

        for t in tokens:
            if t not in {"+", "-", "*", "/"}:
                stack.append(int(t))
                continue

            b = stack.pop()
            a = stack.pop()
            if t == "+":
                stack.append(a + b)
            elif t == "-":
                stack.append(a - b)
            elif t == "*":
                stack.append(a * b)
            else:
                # LeetCode requires truncation toward zero.
                stack.append(int(a / b))

        return stack[-1]


def run_tests() -> None:
    sol = Solution()
    assert sol.evalRPN(["2", "1", "+", "3", "*"]) == 9
    assert sol.evalRPN(["4", "13", "5", "/", "+"]) == 6
    assert sol.evalRPN(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]) == 22
    assert sol.evalRPN(["-3", "2", "/"]) == -1  # trunc toward 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
