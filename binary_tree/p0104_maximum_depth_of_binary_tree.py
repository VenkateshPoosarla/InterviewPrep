"""104. Maximum Depth of Binary Tree

Link: https://leetcode.com/problems/maximum-depth-of-binary-tree/

DFS recursion: depth(node) = 1 + max(depth(left), depth(right)).
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
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))


def run_tests() -> None:
    sol = Solution()

    root = build_tree_level_order([3, 9, 20, None, None, 15, 7])
    assert sol.maxDepth(root) == 3
    assert sol.maxDepth(build_tree_level_order([])) == 0
    assert sol.maxDepth(build_tree_level_order([1])) == 1


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


