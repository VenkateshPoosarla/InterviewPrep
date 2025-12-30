"""841. Keys and Rooms

Link: https://leetcode.com/problems/keys-and-rooms/

We start in room 0, collect keys, and can enter rooms whose key we've collected.
Question: can we visit all rooms?

Graph reachability from node 0 (DFS/BFS).
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List


class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        n = len(rooms)
        if n == 0:
            return True

        seen = [False] * n
        q: Deque[int] = deque([0])
        seen[0] = True

        while q:
            r = q.popleft()
            for nxt in rooms[r]:
                if 0 <= nxt < n and not seen[nxt]:
                    seen[nxt] = True
                    q.append(nxt)

        return all(seen)


def run_tests() -> None:
    sol = Solution()

    assert sol.canVisitAllRooms([[1], [2], [3], []]) is True
    assert sol.canVisitAllRooms([[1, 3], [3, 0, 1], [2], [0]]) is False
    assert sol.canVisitAllRooms([[]]) is True
    assert sol.canVisitAllRooms([]) is True


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


