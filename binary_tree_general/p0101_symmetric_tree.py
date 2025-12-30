"""101. Symmetric Tree

Link: https://leetcode.com/problems/symmetric-tree/

Problem:
Return True if a binary tree is symmetric around its center (mirror of itself).

Approach (mirror DFS):
Two subtrees are mirrors if:
- their roots have equal values
- left.left mirrors right.right
- left.right mirrors right.left

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
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def mirror(a: Optional[TreeNode], b: Optional[TreeNode]) -> bool:
            if a is None and b is None:
                return True
            if a is None or b is None:
                return False
            if a.val != b.val:
                return False
            return mirror(a.left, b.right) and mirror(a.right, b.left)

        return mirror(root.left, root.right) if root is not None else True


def run_tests() -> None:
    sol = Solution()
    assert sol.isSymmetric(build_tree_level_order([1, 2, 2, 3, 4, 4, 3])) is True
    assert sol.isSymmetric(build_tree_level_order([1, 2, 2, None, 3, None, 3])) is False
    assert sol.isSymmetric(None) is True


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
