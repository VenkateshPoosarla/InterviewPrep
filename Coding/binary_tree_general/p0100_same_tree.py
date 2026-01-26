"""100. Same Tree

Link: https://leetcode.com/problems/same-tree/

Problem:
Given the roots of two binary trees, return True if they are structurally identical and
the nodes have the same values.

Approach (DFS recursion):
- If both nodes are None -> equal
- If one is None -> not equal
- Compare values, then compare left subtrees and right subtrees

Complexity:
- Time: O(n) where n is number of nodes (visits each node once)
- Space: O(h) recursion stack, h = tree height
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
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p is None and q is None:
            return True
        if p is None or q is None:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)


def run_tests() -> None:
    sol = Solution()
    assert sol.isSameTree(build_tree_level_order([1, 2, 3]), build_tree_level_order([1, 2, 3])) is True
    assert sol.isSameTree(build_tree_level_order([1, 2]), build_tree_level_order([1, None, 2])) is False
    assert sol.isSameTree(build_tree_level_order([1, 2, 1]), build_tree_level_order([1, 1, 2])) is False


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
