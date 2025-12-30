"""230. Kth Smallest Element in a BST

Link: https://leetcode.com/problems/kth-smallest-element-in-a-bst/

Problem:
Given the root of a BST and an integer k, return the k-th smallest value (1-indexed).

Approach (iterative inorder):
Inorder traversal of BST yields values in sorted order.
Use a stack to traverse left spine, pop nodes, decrement k, and when k hits 0 return node.val.

Complexity:
- Time: O(h + k) average (worst O(n))
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
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack: list[TreeNode] = []
        cur = root
        while cur is not None or stack:
            while cur is not None:
                stack.append(cur)
                cur = cur.left
            node = stack.pop()
            k -= 1
            if k == 0:
                return node.val
            cur = node.right
        raise ValueError("k is out of range")


def run_tests() -> None:
    sol = Solution()
    root = build_tree_level_order([3, 1, 4, None, 2])
    assert sol.kthSmallest(root, 1) == 1
    assert sol.kthSmallest(root, 2) == 2
    assert sol.kthSmallest(root, 3) == 3
    assert sol.kthSmallest(root, 4) == 4

    root = build_tree_level_order([5, 3, 6, 2, 4, None, None, 1])
    assert sol.kthSmallest(root, 3) == 3


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
