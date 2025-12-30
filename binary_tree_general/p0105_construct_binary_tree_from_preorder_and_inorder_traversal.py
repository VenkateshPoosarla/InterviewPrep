"""105. Construct Binary Tree from Preorder and Inorder Traversal

Link: https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/

Problem:
Given preorder and inorder traversal of a binary tree (no duplicates), build the tree.

Approach:
Preorder visits: root, left, right
Inorder visits: left, root, right

Algorithm:
- First element in preorder is the root.
- Find root index in inorder to split left/right subtrees.
- Recurse with inorder boundaries while advancing a preorder index.
Use a hashmap from value -> inorder index for O(1) splits.

Complexity:
- Time: O(n)
- Space: O(n) for hashmap + recursion stack
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
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None

        idx = {v: i for i, v in enumerate(inorder)}
        pre_i = 0

        def build(lo: int, hi: int) -> Optional[TreeNode]:
            nonlocal pre_i
            if lo > hi:
                return None
            root_val = preorder[pre_i]
            pre_i += 1
            mid = idx[root_val]
            root = TreeNode(root_val)
            root.left = build(lo, mid - 1)
            root.right = build(mid + 1, hi)
            return root

        return build(0, len(inorder) - 1)


def run_tests() -> None:
    sol = Solution()
    root = sol.buildTree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7])
    assert tree_to_level_order(root) == [3, 9, 20, None, None, 15, 7]

    root = sol.buildTree([-1], [-1])
    assert tree_to_level_order(root) == [-1]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
