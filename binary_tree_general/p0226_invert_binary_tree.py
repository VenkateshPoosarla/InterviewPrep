"""226. Invert Binary Tree

Link: https://leetcode.com/problems/invert-binary-tree/

Problem:
Invert a binary tree (swap left and right children for every node).

Approach (DFS recursion):
For each node:
- invert left
- invert right
- swap

Complexity:
- Time: O(n)
- Space: O(h) recursion stack
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


def tree_to_level_order(root: Optional[TreeNode]) -> List[Optional[int]]:
    if root is None:
        return []

    out: List[Optional[int]] = []
    q: Deque[Optional[TreeNode]] = deque([root])
    while q:
        node = q.popleft()
        if node is None:
            out.append(None)
            continue
        out.append(node.val)
        q.append(node.left)
        q.append(node.right)

    while out and out[-1] is None:
        out.pop()
    return out


class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return None
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root


def run_tests() -> None:
    sol = Solution()

    root = build_tree_level_order([4, 2, 7, 1, 3, 6, 9])
    out = sol.invertTree(root)
    assert tree_to_level_order(out) == [4, 7, 2, 9, 6, 3, 1]

    root = build_tree_level_order([2, 1, 3])
    out = sol.invertTree(root)
    assert tree_to_level_order(out) == [2, 3, 1]

    assert sol.invertTree(None) is None


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
