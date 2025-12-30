"""222. Count Complete Tree Nodes

Link: https://leetcode.com/problems/count-complete-tree-nodes/

Problem:
Given the root of a complete binary tree, return the number of nodes.

Approach (complete tree height trick):
For a complete tree:
- Compute leftmost height (go left pointers)
- Compute rightmost height (go right pointers)
If heights are equal, it's a perfect tree with \(2^h - 1\) nodes.
Otherwise recurse into children.

Complexity:
- Time: O((log n)^2) (height computation is O(log n) per level)
- Space: O(log n)
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
    def countNodes(self, root: Optional[TreeNode]) -> int:
        def left_h(node: Optional[TreeNode]) -> int:
            h = 0
            while node is not None:
                h += 1
                node = node.left
            return h

        def right_h(node: Optional[TreeNode]) -> int:
            h = 0
            while node is not None:
                h += 1
                node = node.right
            return h

        if root is None:
            return 0
        lh = left_h(root)
        rh = right_h(root)
        if lh == rh:
            return (1 << lh) - 1
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)


def run_tests() -> None:
    sol = Solution()
    assert sol.countNodes(build_tree_level_order([1, 2, 3, 4, 5, 6])) == 6
    assert sol.countNodes(build_tree_level_order([])) == 0
    assert sol.countNodes(build_tree_level_order([1])) == 1


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
