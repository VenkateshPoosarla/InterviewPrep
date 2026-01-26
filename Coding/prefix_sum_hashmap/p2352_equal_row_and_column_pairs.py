"""2352. Equal Row and Column Pairs

Link: https://leetcode.com/problems/equal-row-and-column-pairs/

Count rows as tuples, then for each column tuple add how many times it appeared as a row.

Time: O(n^2)
Space: O(n^2) for row tuples in worst case.
"""

from __future__ import annotations

from collections import Counter
from typing import List


class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        n = len(grid)
        row_counts = Counter(tuple(row) for row in grid)

        ans = 0
        for c in range(n):
            col = tuple(grid[r][c] for r in range(n))
            ans += row_counts[col]
        return ans


def run_tests() -> None:
    sol = Solution()

    assert sol.equalPairs([[3, 2, 1], [1, 7, 6], [2, 7, 7]]) == 1
    assert sol.equalPairs([[3, 1, 2, 2], [1, 4, 4, 5], [2, 4, 2, 2], [2, 4, 2, 2]]) == 3
    assert sol.equalPairs([[1]]) == 1


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


