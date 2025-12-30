"""124. Binary Tree Maximum Path Sum

Link: https://leetcode.com/problems/binary-tree-maximum-path-sum/

Problem:
Return the maximum path sum in a binary tree. A path can start and end at any nodes,
but it must go downwards along parent-child connections (no branching upwards).

Approach (DFS with max gain):
For each node, compute the maximum gain from this node to any descendant:
  gain(node) = node.val + max(0, gain(left), gain(right))

The best path *through* this node (as the highest node on the path) is:
  node.val + max(0, gain(left)) + max(0, gain(right))
Track the maximum of that over all nodes.

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
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        best = -10**18

        def gain(node: Optional[TreeNode]) -> int:
            nonlocal best
            if node is None:
                return 0
            left = max(0, gain(node.left))
            right = max(0, gain(node.right))
            best = max(best, node.val + left + right)
            return node.val + max(left, right)

        gain(root)
        return int(best)


def run_tests() -> None:
    sol = Solution()
    assert sol.maxPathSum(build_tree_level_order([1, 2, 3])) == 6
    assert sol.maxPathSum(build_tree_level_order([-10, 9, 20, None, None, 15, 7])) == 42
    assert sol.maxPathSum(build_tree_level_order([-3])) == -3


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
