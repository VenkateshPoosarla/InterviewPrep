"""79. Word Search

Link: https://leetcode.com/problems/word-search/

Backtracking (DFS) from each cell that matches word[0].
We mark visited cells in-place (temporarily) to avoid revisiting in the current path.
"""

from __future__ import annotations

from typing import List


class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not word:
            return True
        if not board or not board[0]:
            return False

        rows, cols = len(board), len(board[0])

        def dfs(r: int, c: int, i: int) -> bool:
            if i == len(word):
                return True
            if r < 0 or r >= rows or c < 0 or c >= cols:
                return False
            if board[r][c] != word[i]:
                return False

            tmp = board[r][c]
            board[r][c] = "#"  # mark visited
            found = (
                dfs(r + 1, c, i + 1)
                or dfs(r - 1, c, i + 1)
                or dfs(r, c + 1, i + 1)
                or dfs(r, c - 1, i + 1)
            )
            board[r][c] = tmp
            return found

        for r in range(rows):
            for c in range(cols):
                if board[r][c] == word[0] and dfs(r, c, 0):
                    return True
        return False


def run_tests() -> None:
    sol = Solution()

    board = [list("ABCE"), list("SFCS"), list("ADEE")]
    assert sol.exist([row[:] for row in board], "ABCCED") is True
    assert sol.exist([row[:] for row in board], "SEE") is True
    assert sol.exist([row[:] for row in board], "ABCB") is False
    assert sol.exist([list("A")], "A") is True


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


