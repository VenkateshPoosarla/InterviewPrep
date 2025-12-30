"""103. Binary Tree Zigzag Level Order Traversal

Link: https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/

Problem:
Return the zigzag level order traversal of a binary tree:
level 0 left->right, level 1 right->left, alternating.

Approach (BFS + reverse on odd levels):
Perform standard BFS by levels. For odd-indexed levels, reverse the level list before
adding to output.

Complexity:
- Time: O(n)
- Space: O(w)
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
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        res: List[List[int]] = []
        q: Deque[TreeNode] = deque([root])
        left_to_right = True
        while q:
            level: List[int] = []
            for _ in range(len(q)):
                node = q.popleft()
                level.append(node.val)
                if node.left is not None:
                    q.append(node.left)
                if node.right is not None:
                    q.append(node.right)
            if not left_to_right:
                level.reverse()
            res.append(level)
            left_to_right = not left_to_right
        return res


def run_tests() -> None:
    sol = Solution()
    root = build_tree_level_order([3, 9, 20, None, None, 15, 7])
    assert sol.zigzagLevelOrder(root) == [[3], [20, 9], [15, 7]]
    assert sol.zigzagLevelOrder(None) == []


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
