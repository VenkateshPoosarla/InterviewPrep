"""52. N-Queens II

Link: https://leetcode.com/problems/n-queens-ii/

Problem:
Return the number of distinct solutions to the n-queens puzzle.

Approach (backtracking with sets):
Place queens row by row. A position (r,c) is blocked if:
- column c already used
- diag1 (r-c) already used
- diag2 (r+c) already used

Complexity:
- Time: exponential (backtracking)
- Space: O(n) for recursion + sets
"""

from __future__ import annotations

import sys


class Solution:
    def totalNQueens(self, n: int) -> int:
        cols: set[int] = set()
        d1: set[int] = set()  # r - c
        d2: set[int] = set()  # r + c
        count = 0

        def dfs(r: int) -> None:
            nonlocal count
            if r == n:
                count += 1
                return
            for c in range(n):
                if c in cols or (r - c) in d1 or (r + c) in d2:
                    continue
                cols.add(c)
                d1.add(r - c)
                d2.add(r + c)
                dfs(r + 1)
                cols.remove(c)
                d1.remove(r - c)
                d2.remove(r + c)

        dfs(0)
        return count


def run_tests() -> None:
    sol = Solution()
    assert sol.totalNQueens(1) == 1
    assert sol.totalNQueens(4) == 2


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
