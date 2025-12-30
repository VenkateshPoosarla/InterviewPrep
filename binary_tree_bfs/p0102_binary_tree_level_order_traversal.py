"""102. Binary Tree Level Order Traversal

Link: https://leetcode.com/problems/binary-tree-level-order-traversal/

Problem:
Return the level order traversal of a binary tree's nodes' values (left to right, level by level).

Approach (BFS queue):
Standard breadth-first traversal:
- Push root
- For each level, pop all nodes currently in queue and push their children.

Complexity:
- Time: O(n)
- Space: O(w) where w is max width of tree
"""

from __future__ import annotations

import sys
from collections import deque
from typing import Deque, List, Optional

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
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        res: List[List[int]] = []
        q: Deque[TreeNode] = deque([root])
        while q:
            level: List[int] = []
            for _ in range(len(q)):
                node = q.popleft()
                level.append(node.val)
                if node.left is not None:
                    q.append(node.left)
                if node.right is not None:
                    q.append(node.right)
            res.append(level)
        return res


def run_tests() -> None:
    sol = Solution()
    assert sol.levelOrder(build_tree_level_order([3, 9, 20, None, None, 15, 7])) == [[3], [9, 20], [15, 7]]
    assert sol.levelOrder(build_tree_level_order([1])) == [[1]]
    assert sol.levelOrder(None) == []


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
