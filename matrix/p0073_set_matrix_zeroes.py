"""73. Set Matrix Zeroes

Link: https://leetcode.com/problems/set-matrix-zeroes/

Problem:
If an element is 0, set its entire row and column to 0, in-place.

Approach (use first row/col as markers):
We need O(1) extra space.
1) Determine if first row / first column originally contain any zeros.
2) For the rest of the matrix, if matrix[r][c] == 0, mark:
   - matrix[r][0] = 0 (row marker)
   - matrix[0][c] = 0 (col marker)
3) Zero out cells based on these markers.
4) Finally, zero out first row/col if they originally had zeros.

Complexity:
- Time: O(m*n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        if not matrix or not matrix[0]:
            return
        m, n = len(matrix), len(matrix[0])

        first_row_zero = any(matrix[0][c] == 0 for c in range(n))
        first_col_zero = any(matrix[r][0] == 0 for r in range(m))

        for r in range(1, m):
            for c in range(1, n):
                if matrix[r][c] == 0:
                    matrix[r][0] = 0
                    matrix[0][c] = 0

        for r in range(1, m):
            for c in range(1, n):
                if matrix[r][0] == 0 or matrix[0][c] == 0:
                    matrix[r][c] = 0

        if first_row_zero:
            for c in range(n):
                matrix[0][c] = 0
        if first_col_zero:
            for r in range(m):
                matrix[r][0] = 0


def run_tests() -> None:
    sol = Solution()

    m = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    sol.setZeroes(m)
    assert m == [[1, 0, 1], [0, 0, 0], [1, 0, 1]]

    m = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]
    sol.setZeroes(m)
    assert m == [[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
