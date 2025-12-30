"""1448. Count Good Nodes in Binary Tree

Link: https://leetcode.com/problems/count-good-nodes-in-binary-tree/

A node is "good" if on the path from root to it, no value is greater than it.
DFS while tracking max_so_far on the path.
"""

from __future__ import annotations

import sys
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
    def goodNodes(self, root: Optional[TreeNode]) -> int:
        def dfs(node: Optional[TreeNode], max_so_far: int) -> int:
            if node is None:
                return 0
            good = 1 if node.val >= max_so_far else 0
            max2 = max(max_so_far, node.val)
            return good + dfs(node.left, max2) + dfs(node.right, max2)

        if root is None:
            return 0
        return dfs(root, root.val)


def run_tests() -> None:
    sol = Solution()

    root = build_tree_level_order([3, 1, 4, 3, None, 1, 5])
    assert sol.goodNodes(root) == 4

    root = build_tree_level_order([3, 3, None, 4, 2])
    assert sol.goodNodes(root) == 3

    root = build_tree_level_order([1])
    assert sol.goodNodes(root) == 1


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


