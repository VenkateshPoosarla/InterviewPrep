"""48. Rotate Image

Link: https://leetcode.com/problems/rotate-image/

Problem:
Rotate an n x n matrix 90 degrees clockwise in-place.

Approach (transpose + reverse rows):
Clockwise rotation can be done by:
1) Transpose: swap matrix[r][c] with matrix[c][r] for r < c
2) Reverse each row

Complexity:
- Time: O(n^2)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        # transpose
        for r in range(n):
            for c in range(r + 1, n):
                matrix[r][c], matrix[c][r] = matrix[c][r], matrix[r][c]
        # reverse rows
        for r in range(n):
            matrix[r].reverse()


def run_tests() -> None:
    sol = Solution()

    m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    sol.rotate(m)
    assert m == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]

    m = [[1, 2], [3, 4]]
    sol.rotate(m)
    assert m == [[3, 1], [4, 2]]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
