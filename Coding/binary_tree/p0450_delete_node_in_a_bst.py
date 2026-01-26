"""450. Delete Node in a BST

Link: https://leetcode.com/problems/delete-node-in-a-bst/

Standard BST deletion:
- If key < root.val: delete in left subtree
- If key > root.val: delete in right subtree
- Else found node:
  - if one child missing: return the other
  - if two children: replace value with inorder successor (min of right subtree),
    then delete successor node from right subtree.
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
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if root is None:
            return None
        if key < root.val:
            root.left = self.deleteNode(root.left, key)
            return root
        if key > root.val:
            root.right = self.deleteNode(root.right, key)
            return root

        # key == root.val
        if root.left is None:
            return root.right
        if root.right is None:
            return root.left

        # Two children: replace with inorder successor.
        succ = root.right
        while succ.left is not None:
            succ = succ.left
        root.val = succ.val
        root.right = self.deleteNode(root.right, succ.val)
        return root


def _inorder(root: Optional[TreeNode]) -> List[int]:
    if root is None:
        return []
    return _inorder(root.left) + [root.val] + _inorder(root.right)


def run_tests() -> None:
    sol = Solution()

    root = build_tree_level_order([5, 3, 6, 2, 4, None, 7])
    out = sol.deleteNode(root, 3)
    assert _inorder(out) == [2, 4, 5, 6, 7]

    root = build_tree_level_order([5, 3, 6, 2, 4, None, 7])
    out = sol.deleteNode(root, 0)
    assert _inorder(out) == [2, 3, 4, 5, 6, 7]

    root = build_tree_level_order([5, 3, 6, 2, 4, None, 7])
    out = sol.deleteNode(root, 5)
    assert _inorder(out) == [2, 3, 4, 6, 7]

    root = build_tree_level_order([1])
    out = sol.deleteNode(root, 1)
    assert _inorder(out) == []


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


