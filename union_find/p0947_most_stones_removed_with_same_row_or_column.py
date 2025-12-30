"""947. Most Stones Removed with Same Row or Column

Link: https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/

Pattern: Union-Find (DSU) with bipartite modeling (rows <-> cols).

Model each stone (r, c) as an edge between a "row node" r and a "col node" c.
In each connected component, you can remove all but 1 stone.
Answer = total_stones - number_of_connected_components_over_used_nodes.
"""

from __future__ import annotations

import sys
from typing import Dict, List


class DSU:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}
        self.size: Dict[int, int] = {}

    def _add(self, x: int) -> None:
        if x in self.parent:
            return
        self.parent[x] = x
        self.size[x] = 1

    def find(self, x: int) -> int:
        self._add(x)
        # path compression
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        dsu = DSU()
        used = set()
        # offset cols so they don't collide with row ids
        COL_OFFSET = 10_000
        for r, c in stones:
            rr = int(r)
            cc = COL_OFFSET + int(c)
            dsu.union(rr, cc)
            used.add(rr)
            used.add(cc)

        comps = {dsu.find(x) for x in used}
        return len(stones) - len(comps)


def run_tests() -> None:
    sol = Solution()
    assert sol.removeStones([[0, 0], [0, 1], [1, 0], [1, 2], [2, 1], [2, 2]]) == 5
    assert sol.removeStones([[0, 0], [0, 2], [1, 1], [2, 0], [2, 2]]) == 3
    assert sol.removeStones([[0, 0]]) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


