"""112. Path Sum

Link: https://leetcode.com/problems/path-sum/

Problem:
Return True if the tree has a root-to-leaf path such that adding up all the values
along the path equals targetSum.

Approach (DFS):
At each node, subtract node.val from target and recurse.
When at a leaf, check whether remaining target equals node.val.

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
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if root is None:
            return False
        if root.left is None and root.right is None:
            return root.val == targetSum
        remaining = targetSum - root.val
        return self.hasPathSum(root.left, remaining) or self.hasPathSum(root.right, remaining)


def run_tests() -> None:
    sol = Solution()
    root = build_tree_level_order([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1])
    assert sol.hasPathSum(root, 22) is True
    root = build_tree_level_order([1, 2, 3])
    assert sol.hasPathSum(root, 5) is False
    assert sol.hasPathSum(None, 0) is False


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
