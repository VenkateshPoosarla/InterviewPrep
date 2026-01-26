"""20. Valid Parentheses

Link: https://leetcode.com/problems/valid-parentheses/

Problem:
Given a string containing just the characters '()[]{}', determine if it is valid:
- Open brackets must be closed by the same type.
- Open brackets must be closed in the correct order.

Approach (stack):
Push opening brackets. For a closing bracket, the stack top must be the matching opener.

Complexity:
- Time: O(n)
- Space: O(n)
"""

from __future__ import annotations

import sys


class Solution:
    def isValid(self, s: str) -> bool:
        match = {")": "(", "]": "[", "}": "{"}
        stack: list[str] = []
        for ch in s:
            if ch in match.values():
                stack.append(ch)
            else:
                if not stack or stack[-1] != match.get(ch):
                    return False
                stack.pop()
        return not stack


def run_tests() -> None:
    sol = Solution()
    assert sol.isValid("()") is True
    assert sol.isValid("()[]{}") is True
    assert sol.isValid("(]") is False
    assert sol.isValid("([)]") is False
    assert sol.isValid("{[]}") is True
    assert sol.isValid("") is True


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
