"""289. Game of Life

Link: https://leetcode.com/problems/game-of-life/

Problem:
Update the board to the next state in Conway's Game of Life, in-place.
Rules per cell (live=1, dead=0):
- Any live cell with <2 live neighbors dies.
- Any live cell with 2 or 3 live neighbors lives.
- Any live cell with >3 live neighbors dies.
- Any dead cell with exactly 3 live neighbors becomes live.

Approach (in-place with state encoding):
We need neighbors computed from the *original* state while writing the new state.
Encode transitions in the same grid:
- 0: dead -> dead
- 1: live -> live
- 2: live -> dead  (was live, will die)
- 3: dead -> live  (was dead, will become live)
When counting live neighbors, treat cells with original live state as live: values 1 or 2.
At the end, reduce each cell to new state: cell % 2 (or (cell==1 or cell==3)).

Complexity:
- Time: O(m*n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        if not board or not board[0]:
            return
        m, n = len(board), len(board[0])

        def live_neighbors(r: int, c: int) -> int:
            cnt = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < m and 0 <= cc < n:
                        if board[rr][cc] in (1, 2):
                            cnt += 1
            return cnt

        for r in range(m):
            for c in range(n):
                ln = live_neighbors(r, c)
                if board[r][c] == 1:
                    if ln < 2 or ln > 3:
                        board[r][c] = 2  # live -> dead
                else:
                    if ln == 3:
                        board[r][c] = 3  # dead -> live

        for r in range(m):
            for c in range(n):
                board[r][c] %= 2


def run_tests() -> None:
    sol = Solution()

    b = [[0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]
    sol.gameOfLife(b)
    assert b == [[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0]]

    b = [[1, 1], [1, 0]]
    sol.gameOfLife(b)
    assert b == [[1, 1], [1, 1]]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
