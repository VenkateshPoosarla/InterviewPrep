"""872. Leaf-Similar Trees

Link: https://leetcode.com/problems/leaf-similar-trees/

Collect leaf values (left-to-right) for each tree and compare sequences.
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
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        return self._leaves(root1) == self._leaves(root2)

    def _leaves(self, root: Optional[TreeNode]) -> List[int]:
        out: List[int] = []

        def dfs(node: Optional[TreeNode]) -> None:
            if node is None:
                return
            if node.left is None and node.right is None:
                out.append(node.val)
                return
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return out


def run_tests() -> None:
    sol = Solution()

    # LeetCode example
    t1 = build_tree_level_order([3, 5, 1, 6, 2, 9, 8, None, None, 7, 4])
    t2 = build_tree_level_order([3, 5, 1, 6, 7, 4, 2, None, None, None, None, None, None, 9, 8])
    assert sol.leafSimilar(t1, t2) is True

    t3 = build_tree_level_order([1, 2, 3])
    t4 = build_tree_level_order([1, 3, 2])
    assert sol.leafSimilar(t3, t4) is False


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


