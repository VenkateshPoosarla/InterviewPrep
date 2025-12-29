"""54. Spiral Matrix

Link: https://leetcode.com/problems/spiral-matrix/

Problem:
Return all elements of the matrix in spiral order.

Approach (shrink boundaries):
Maintain four boundaries:
- top, bottom row indices
- left, right column indices
Traverse:
1) top row left->right, top++
2) right col top->bottom, right--
3) bottom row right->left, bottom--
4) left col bottom->top, left++
Stop when boundaries cross.

Complexity:
- Time: O(m*n)
- Space: O(1) extra (output excluded)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix or not matrix[0]:
            return []

        top, bottom = 0, len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1
        out: List[int] = []

        while top <= bottom and left <= right:
            for c in range(left, right + 1):
                out.append(matrix[top][c])
            top += 1

            for r in range(top, bottom + 1):
                out.append(matrix[r][right])
            right -= 1

            if top <= bottom:
                for c in range(right, left - 1, -1):
                    out.append(matrix[bottom][c])
                bottom -= 1

            if left <= right:
                for r in range(bottom, top - 1, -1):
                    out.append(matrix[r][left])
                left += 1

        return out


def run_tests() -> None:
    sol = Solution()
    assert sol.spiralOrder([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]
    assert sol.spiralOrder([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) == [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
    assert sol.spiralOrder([[1]]) == [1]
    assert sol.spiralOrder([[]]) == []


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
