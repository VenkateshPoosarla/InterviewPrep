"""36. Valid Sudoku

Link: https://leetcode.com/problems/valid-sudoku/

Problem:
Determine if a 9x9 Sudoku board is valid (partially filled is allowed).
Rules:
- Each row contains digits 1-9 at most once.
- Each column contains digits 1-9 at most once.
- Each 3x3 sub-box contains digits 1-9 at most once.
Empty cells are '.'.

Approach:
Use sets to track seen digits:
- rows[r], cols[c], boxes[b] where b = (r//3)*3 + (c//3)
If a digit repeats in any set, invalid.

Complexity:
- Time: O(81) = O(1)
- Space: O(81) = O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]

        for r in range(9):
            for c in range(9):
                ch = board[r][c]
                if ch == ".":
                    continue
                b = (r // 3) * 3 + (c // 3)
                if ch in rows[r] or ch in cols[c] or ch in boxes[b]:
                    return False
                rows[r].add(ch)
                cols[c].add(ch)
                boxes[b].add(ch)
        return True


def run_tests() -> None:
    sol = Solution()

    board = [
        ["5", "3", ".", ".", "7", ".", ".", ".", "."],
        ["6", ".", ".", "1", "9", "5", ".", ".", "."],
        [".", "9", "8", ".", ".", ".", ".", "6", "."],
        ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
        ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
        ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
        [".", "6", ".", ".", ".", ".", "2", "8", "."],
        [".", ".", ".", "4", "1", "9", ".", ".", "5"],
        [".", ".", ".", ".", "8", ".", ".", "7", "9"],
    ]
    assert sol.isValidSudoku(board) is True

    bad = [
        ["8", "3", ".", ".", "7", ".", ".", ".", "."],
        ["6", ".", ".", "1", "9", "5", ".", ".", "."],
        [".", "9", "8", ".", ".", ".", ".", "6", "."],
        ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
        ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
        ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
        [".", "6", ".", ".", ".", ".", "2", "8", "."],
        [".", ".", ".", "4", "1", "9", ".", ".", "5"],
        [".", ".", ".", ".", "8", ".", ".", "7", "9"],
    ]
    assert sol.isValidSudoku(bad) is False


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
