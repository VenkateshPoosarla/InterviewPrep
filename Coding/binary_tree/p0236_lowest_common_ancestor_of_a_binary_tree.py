"""236. Lowest Common Ancestor of a Binary Tree

Link: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/

Recursive:
- If current node is None or equals p or q, return it.
- Recurse into left/right.
- If both sides return non-null, current node is LCA.
- Else return the non-null side.
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
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if root is None or root is p or root is q:
            return root

        left = self.lowestCommonAncestor(root.left, p, q) if root.left else None
        right = self.lowestCommonAncestor(root.right, p, q) if root.right else None

        if left and right:
            return root
        return left if left else right  # type: ignore[return-value]


def _find_node(root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    if root is None:
        return None
    if root.val == val:
        return root
    return _find_node(root.left, val) or _find_node(root.right, val)


def run_tests() -> None:
    sol = Solution()

    root = build_tree_level_order([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
    p = _find_node(root, 5)
    q = _find_node(root, 1)
    assert p is not None and q is not None and root is not None
    assert sol.lowestCommonAncestor(root, p, q).val == 3

    q2 = _find_node(root, 4)
    assert q2 is not None
    assert sol.lowestCommonAncestor(root, p, q2).val == 5


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


