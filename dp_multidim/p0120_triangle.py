"""120. Triangle

Link: https://leetcode.com/problems/triangle/

Problem:
Given a triangle array, return the minimum path sum from top to bottom.
At each step, you may move to adjacent numbers on the row below.

Approach (bottom-up DP, O(n) space):
Start from the last row as dp. For each row upwards:
  dp[c] = triangle[r][c] + min(dp[c], dp[c+1])
After finishing, dp[0] is the answer.

Complexity:
- Time: O(n^2) over all cells
- Space: O(n) where n is row length
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if not triangle:
            return 0
        dp = triangle[-1][:]  # copy last row
        for r in range(len(triangle) - 2, -1, -1):
            for c in range(r + 1):
                dp[c] = triangle[r][c] + min(dp[c], dp[c + 1])
        return dp[0]


def run_tests() -> None:
    sol = Solution()
    assert sol.minimumTotal([[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]) == 11
    assert sol.minimumTotal([[-10]]) == -10


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
