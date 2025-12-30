"""98. Validate Binary Search Tree

Link: https://leetcode.com/problems/validate-binary-search-tree/

Problem:
Return True if a binary tree is a valid BST:
- left subtree values < node.val
- right subtree values > node.val
- both subtrees must also be BSTs

Approach (DFS with bounds):
Carry (low, high) bounds down the recursion:
- node.val must satisfy low < node.val < high
- left uses (low, node.val)
- right uses (node.val, high)

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
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def dfs(node: Optional[TreeNode], low: float, high: float) -> bool:
            if node is None:
                return True
            if not (low < node.val < high):
                return False
            return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)

        return dfs(root, float("-inf"), float("inf"))


def run_tests() -> None:
    sol = Solution()
    assert sol.isValidBST(build_tree_level_order([2, 1, 3])) is True
    assert sol.isValidBST(build_tree_level_order([5, 1, 4, None, None, 3, 6])) is False
    assert sol.isValidBST(build_tree_level_order([2, 2, 2])) is False


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
