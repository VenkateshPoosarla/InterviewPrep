"""427. Construct Quad Tree

Link: https://leetcode.com/problems/construct-quad-tree/

Problem:
Given an n x n binary matrix grid, construct a quad tree.
Each node represents a region. If all values in the region are the same, it's a leaf.

Approach (divide and conquer):
Recursively split region into 4 quadrants until it is uniform.
If 4 children are leaves with the same value, compress into one leaf node.

Complexity:
- Time: O(n^2) typical
- Space: O(n^2) for recursion/tree nodes worst case
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Node:
    val: bool
    isLeaf: bool
    topLeft: Optional["Node"] = None
    topRight: Optional["Node"] = None
    bottomLeft: Optional["Node"] = None
    bottomRight: Optional["Node"] = None


class Solution:
    def construct(self, grid: List[List[int]]) -> Optional[Node]:
        if not grid or not grid[0]:
            return None
        n = len(grid)

        def build(r0: int, c0: int, size: int) -> Node:
            first = grid[r0][c0]
            uniform = True
            for r in range(r0, r0 + size):
                for c in range(c0, c0 + size):
                    if grid[r][c] != first:
                        uniform = False
                        break
                if not uniform:
                    break
            if uniform:
                return Node(val=bool(first), isLeaf=True)

            half = size // 2
            tl = build(r0, c0, half)
            tr = build(r0, c0 + half, half)
            bl = build(r0 + half, c0, half)
            br = build(r0 + half, c0 + half, half)

            if tl.isLeaf and tr.isLeaf and bl.isLeaf and br.isLeaf and {tl.val, tr.val, bl.val, br.val}.__len__() == 1:
                return Node(val=tl.val, isLeaf=True)

            return Node(val=True, isLeaf=False, topLeft=tl, topRight=tr, bottomLeft=bl, bottomRight=br)

        return build(0, 0, n)


def run_tests() -> None:
    sol = Solution()
    grid = [
        [0, 1],
        [1, 0],
    ]
    root = sol.construct(grid)
    assert root is not None and root.isLeaf is False
    assert root.topLeft is not None and root.topLeft.isLeaf is True and root.topLeft.val is False
    assert root.topRight is not None and root.topRight.isLeaf is True and root.topRight.val is True

    grid = [
        [1, 1],
        [1, 1],
    ]
    root = sol.construct(grid)
    assert root is not None and root.isLeaf is True and root.val is True


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
