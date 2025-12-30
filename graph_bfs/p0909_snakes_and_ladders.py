"""909. Snakes and Ladders

Link: https://leetcode.com/problems/snakes-and-ladders/

Problem:
Given an n x n board representing a snakes and ladders game, return the minimum number
of moves to reach square n^2 from square 1. If impossible, return -1.

Board numbering:
Squares are labeled from 1 to n^2 in a boustrophedon pattern starting from the bottom-left.
board[r][c] == -1 means normal square, otherwise it contains a destination square
for a snake/ladder.

Approach (BFS on squares):
Each move rolls 1..6. From square x, you can go to x+1..x+6 (<= n^2), then take snake/ladder
if present. BFS yields the minimum moves.

Complexity:
- Time: O(n^2)
- Space: O(n^2)
"""

from __future__ import annotations

import sys
from collections import deque
from typing import Deque, List


class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        n = len(board)

        def to_rc(square: int) -> tuple[int, int]:
            # 1-indexed square -> (r,c) in board
            q, rem = divmod(square - 1, n)
            r = n - 1 - q
            # row direction alternates
            if q % 2 == 0:
                c = rem
            else:
                c = n - 1 - rem
            return r, c

        target = n * n
        q: Deque[tuple[int, int]] = deque([(1, 0)])
        seen = {1}

        while q:
            square, moves = q.popleft()
            if square == target:
                return moves
            for nxt in range(square + 1, min(target, square + 6) + 1):
                r, c = to_rc(nxt)
                dest = board[r][c]
                final = dest if dest != -1 else nxt
                if final not in seen:
                    seen.add(final)
                    q.append((final, moves + 1))
        return -1


def run_tests() -> None:
    sol = Solution()
    board = [
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, 35, -1, -1, 13, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, 15, -1, -1, -1, -1],
    ]
    assert sol.snakesAndLadders(board) == 4

    board = [[-1, -1], [-1, 3]]
    assert sol.snakesAndLadders(board) == 1


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
