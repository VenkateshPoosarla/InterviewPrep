"""221. Maximal Square

Link: https://leetcode.com/problems/maximal-square/

Problem:
Given an m x n binary matrix filled with '0' and '1', find the largest square containing
only '1's and return its area.

Approach (DP with rolling row):
Let dp[c] be the side length of the largest square ending at current row, column c-1
(1-indexed dp to simplify borders). For cell (r,c):
If matrix[r][c] == '1':
  dp[c] = 1 + min(dp[c] (top), dp[c-1] (left), prev_diag (top-left))
Else dp[c] = 0

Complexity:
- Time: O(m*n)
- Space: O(n)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0
        m, n = len(matrix), len(matrix[0])
        dp = [0] * (n + 1)
        best = 0

        for r in range(1, m + 1):
            prev_diag = 0
            for c in range(1, n + 1):
                temp = dp[c]
                if matrix[r - 1][c - 1] == "1":
                    dp[c] = 1 + min(dp[c], dp[c - 1], prev_diag)
                    best = max(best, dp[c])
                else:
                    dp[c] = 0
                prev_diag = temp

        return best * best


def run_tests() -> None:
    sol = Solution()
    assert sol.maximalSquare(
        [["1", "0", "1", "0", "0"], ["1", "0", "1", "1", "1"], ["1", "1", "1", "1", "1"], ["1", "0", "0", "1", "0"]]
    ) == 4
    assert sol.maximalSquare([["0", "1"], ["1", "0"]]) == 1
    assert sol.maximalSquare([["0"]]) == 0
    assert sol.maximalSquare([["1"]]) == 1


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
