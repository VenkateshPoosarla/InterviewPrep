"""22. Generate Parentheses

Link: https://leetcode.com/problems/generate-parentheses/

Problem:
Given n pairs of parentheses, generate all combinations of well-formed parentheses.

Approach (backtracking with constraints):
Build the string by choosing '(' or ')' as long as:
- open_used < n  (we can still add '(')
- close_used < open_used (we can only close if there's an unmatched '(')

Complexity:
- Time: O(Cn) where Cn is the n-th Catalan number
- Space: O(n) recursion depth + output
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res: List[str] = []
        path: list[str] = []

        def dfs(open_used: int, close_used: int) -> None:
            if open_used == n and close_used == n:
                res.append("".join(path))
                return
            if open_used < n:
                path.append("(")
                dfs(open_used + 1, close_used)
                path.pop()
            if close_used < open_used:
                path.append(")")
                dfs(open_used, close_used + 1)
                path.pop()

        dfs(0, 0)
        return res


def run_tests() -> None:
    sol = Solution()
    assert sorted(sol.generateParenthesis(3)) == sorted(["((()))", "(()())", "(())()", "()(())", "()()()"])
    assert sol.generateParenthesis(1) == ["()"]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
