"""530. Minimum Absolute Difference in BST

Link: https://leetcode.com/problems/minimum-absolute-difference-in-bst/

Problem:
Given the root of a BST, return the minimum absolute difference between values of any
two different nodes.

Approach (inorder):
Inorder traversal visits BST values in sorted order. The minimum difference must be
between two adjacent values in this sorted order.

Track `prev` value and update `best` as we traverse.

Complexity:
- Time: O(n)
- Space: O(h)
"""

from __future__ import annotations

import sys
from typing import Optional

from collections import deque
from typing import Deque, List, Optional


class TreeNode:
    def __init__(
        self,
        val: int = 0,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
    ):
        self.val = int(val)
        self.left = left
        self.right = right


def build_tree_level_order(values: List[Optional[int]]) -> Optional[TreeNode]:
    if not values or values[0] is None:
        return None

    root = TreeNode(values[0])
    q: Deque[TreeNode] = deque([root])
    i = 1
    while q and i < len(values):
        node = q.popleft()

        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            q.append(node.left)
        i += 1

        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            q.append(node.right)
        i += 1

    return root


class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        prev: Optional[int] = None
        best = float("inf")

        def inorder(node: Optional[TreeNode]) -> None:
            nonlocal prev, best
            if node is None:
                return
            inorder(node.left)
            if prev is not None:
                best = min(best, node.val - prev)
            prev = node.val
            inorder(node.right)

        inorder(root)
        return int(best)


def run_tests() -> None:
    sol = Solution()
    assert sol.getMinimumDifference(build_tree_level_order([4, 2, 6, 1, 3])) == 1
    assert sol.getMinimumDifference(build_tree_level_order([1, 0, 48, None, None, 12, 49])) == 1


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
