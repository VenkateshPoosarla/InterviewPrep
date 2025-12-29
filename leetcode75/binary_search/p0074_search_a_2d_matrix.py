"""74. Search a 2D Matrix

Link: https://leetcode.com/problems/search-a-2d-matrix/

Matrix properties:
- each row sorted
- first element of each row > last element of previous row

So treat it as a flattened sorted array of length m*n and binary search.
Index mapping:
  idx -> (r = idx // n, c = idx % n)
"""

from __future__ import annotations

from typing import List


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix or not matrix[0]:
            return False
        m, n = len(matrix), len(matrix[0])
        lo, hi = 0, m * n - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            r, c = divmod(mid, n)
            v = matrix[r][c]
            if v == target:
                return True
            if v < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return False


def run_tests() -> None:
    sol = Solution()
    mat = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]
    assert sol.searchMatrix(mat, 3) is True
    assert sol.searchMatrix(mat, 13) is False
    assert sol.searchMatrix([], 1) is False
    assert sol.searchMatrix([[]], 1) is False


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


