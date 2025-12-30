"""130. Surrounded Regions

Link: https://leetcode.com/problems/surrounded-regions/

Problem:
Given an m x n board containing 'X' and 'O', capture all regions surrounded by 'X'.
An 'O' region is captured by flipping all 'O's into 'X's in that surrounded region.
'O's connected to the border cannot be captured.

Approach (mark border-connected 'O's):
1) BFS/DFS from all border cells that are 'O', marking them as safe (e.g. '#').
2) Flip remaining 'O' -> 'X' (captured).
3) Flip '#' -> 'O' (restore safe).

Complexity:
- Time: O(m*n)
- Space: O(m*n) worst-case queue/stack
"""

from __future__ import annotations

import sys
from collections import deque
from typing import Deque, List


class Solution:
    def solve(self, board: List[List[str]]) -> None:
        if not board or not board[0]:
            return
        m, n = len(board), len(board[0])

        q: Deque[tuple[int, int]] = deque()

        def add_if_o(r: int, c: int) -> None:
            if 0 <= r < m and 0 <= c < n and board[r][c] == "O":
                board[r][c] = "#"
                q.append((r, c))

        # enqueue border O's
        for c in range(n):
            add_if_o(0, c)
            add_if_o(m - 1, c)
        for r in range(m):
            add_if_o(r, 0)
            add_if_o(r, n - 1)

        # BFS to mark all border-connected O's
        while q:
            r, c = q.popleft()
            add_if_o(r + 1, c)
            add_if_o(r - 1, c)
            add_if_o(r, c + 1)
            add_if_o(r, c - 1)

        # finalize flips
        for r in range(m):
            for c in range(n):
                if board[r][c] == "O":
                    board[r][c] = "X"
                elif board[r][c] == "#":
                    board[r][c] = "O"


def run_tests() -> None:
    sol = Solution()
    board = [
        ["X", "X", "X", "X"],
        ["X", "O", "O", "X"],
        ["X", "X", "O", "X"],
        ["X", "O", "X", "X"],
    ]
    sol.solve(board)
    assert board == [
        ["X", "X", "X", "X"],
        ["X", "X", "X", "X"],
        ["X", "X", "X", "X"],
        ["X", "O", "X", "X"],
    ]

    board = [["X"]]
    sol.solve(board)
    assert board == [["X"]]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
