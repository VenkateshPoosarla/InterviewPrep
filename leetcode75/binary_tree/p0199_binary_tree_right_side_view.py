"""199. Binary Tree Right Side View

Link: https://leetcode.com/problems/binary-tree-right-side-view/

Level-order traversal (BFS):
For each level, the rightmost node is visible.
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
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []

        q: Deque[TreeNode] = deque([root])
        out: List[int] = []

        while q:
            level_size = len(q)
            last_val = 0
            for _ in range(level_size):
                node = q.popleft()
                last_val = node.val
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            out.append(last_val)

        return out


def run_tests() -> None:
    sol = Solution()

    root = build_tree_level_order([1, 2, 3, None, 5, None, 4])
    assert sol.rightSideView(root) == [1, 3, 4]

    assert sol.rightSideView(build_tree_level_order([1, None, 3])) == [1, 3]
    assert sol.rightSideView(build_tree_level_order([])) == []


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


