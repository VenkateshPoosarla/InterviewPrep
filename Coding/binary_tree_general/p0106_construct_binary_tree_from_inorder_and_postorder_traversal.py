"""106. Construct Binary Tree from Inorder and Postorder Traversal

Link: https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/

Problem:
Given inorder and postorder traversal arrays of a binary tree (no duplicates), build the tree.

Approach:
Postorder: left, right, root -> root is the last element.
Inorder: left, root, right -> root splits left/right subtrees.

When consuming postorder from the end, we must build the RIGHT subtree first (because
postorder ends with ... left-subtree, right-subtree, root).

Complexity:
- Time: O(n)
- Space: O(n) for index map + recursion stack
"""

from __future__ import annotations

import sys
from typing import List, Optional

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
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not inorder or not postorder:
            return None

        idx = {v: i for i, v in enumerate(inorder)}
        post_i = len(postorder) - 1

        def build(lo: int, hi: int) -> Optional[TreeNode]:
            nonlocal post_i
            if lo > hi:
                return None
            root_val = postorder[post_i]
            post_i -= 1
            mid = idx[root_val]
            root = TreeNode(root_val)
            root.right = build(mid + 1, hi)
            root.left = build(lo, mid - 1)
            return root

        return build(0, len(inorder) - 1)


def run_tests() -> None:
    sol = Solution()
    root = sol.buildTree([9, 3, 15, 20, 7], [9, 15, 7, 20, 3])
    assert tree_to_level_order(root) == [3, 9, 20, None, None, 15, 7]

    root = sol.buildTree([-1], [-1])
    assert tree_to_level_order(root) == [-1]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
