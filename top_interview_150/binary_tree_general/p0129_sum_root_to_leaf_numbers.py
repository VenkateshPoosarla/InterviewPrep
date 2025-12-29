"""129. Sum Root to Leaf Numbers

Link: https://leetcode.com/problems/sum-root-to-leaf-numbers/

Problem:
Each root-to-leaf path represents a number formed by concatenating node values (digits).
Return the total sum of all root-to-leaf numbers.

Approach (DFS accumulate):
Carry the number formed so far:
  next_val = current * 10 + node.val
When we reach a leaf, contribute next_val.

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
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def dfs(node: Optional[TreeNode], cur: int) -> int:
            if node is None:
                return 0
            nxt = cur * 10 + node.val
            if node.left is None and node.right is None:
                return nxt
            return dfs(node.left, nxt) + dfs(node.right, nxt)

        return dfs(root, 0)


def run_tests() -> None:
    sol = Solution()
    assert sol.sumNumbers(build_tree_level_order([1, 2, 3])) == 25  # 12 + 13
    assert sol.sumNumbers(build_tree_level_order([4, 9, 0, 5, 1])) == 1026  # 495 + 491 + 40
    assert sol.sumNumbers(None) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
