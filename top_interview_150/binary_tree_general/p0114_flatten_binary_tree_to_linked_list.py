"""114. Flatten Binary Tree to Linked List

Link: https://leetcode.com/problems/flatten-binary-tree-to-linked-list/

Problem:
Flatten the tree into a "linked list" in-place following preorder traversal:
- right pointers form the list
- left pointers must be None

Approach (reverse preorder recursion):
If we process nodes in reverse preorder (right, left, root), we can keep a `prev` pointer
to the already-flattened part and link:
  node.right = prev
  node.left = None
  prev = node

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
    def flatten(self, root: Optional[TreeNode]) -> None:
        prev: Optional[TreeNode] = None

        def dfs(node: Optional[TreeNode]) -> None:
            nonlocal prev
            if node is None:
                return
            dfs(node.right)
            dfs(node.left)
            node.right = prev
            node.left = None
            prev = node

        dfs(root)


def run_tests() -> None:
    sol = Solution()
    root = build_tree_level_order([1, 2, 5, 3, 4, None, 6])
    sol.flatten(root)
    # flattened list is 1-2-3-4-5-6 (right pointers)
    assert tree_to_level_order(root) == [1, None, 2, None, 3, None, 4, None, 5, None, 6]

    root = build_tree_level_order([])
    sol.flatten(root)
    assert root is None


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
