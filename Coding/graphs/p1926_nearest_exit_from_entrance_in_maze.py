"""1926. Nearest Exit from Entrance in Maze

Link: https://leetcode.com/problems/nearest-exit-from-entrance-in-maze/

BFS from entrance over open cells '.'.
The nearest exit is any open cell on the boundary (excluding entrance itself).
Return minimum steps or -1 if none.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple


class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        rows = len(maze)
        cols = len(maze[0]) if rows else 0
        if rows == 0 or cols == 0:
            return -1

        sr, sc = entrance
        q: Deque[Tuple[int, int, int]] = deque([(sr, sc, 0)])
        seen = [[False] * cols for _ in range(rows)]
        seen[sr][sc] = True

        def is_exit(r: int, c: int) -> bool:
            if r == sr and c == sc:
                return False
            return r == 0 or c == 0 or r == rows - 1 or c == cols - 1

        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while q:
            r, c, d = q.popleft()
            if is_exit(r, c):
                return d
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not seen[nr][nc] and maze[nr][nc] == ".":
                    seen[nr][nc] = True
                    q.append((nr, nc, d + 1))

        return -1


def run_tests() -> None:
    sol = Solution()

    maze = [list("+.+"), list("..."), list("+.+")]
    assert sol.nearestExit(maze, [1, 0]) == 2

    maze = [list("+.+"), list(".+."), list("...")]
    assert sol.nearestExit(maze, [2, 0]) == 1

    maze = [list("+++"), list("+.+"), list("+++")]
    assert sol.nearestExit(maze, [1, 1]) == -1


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


