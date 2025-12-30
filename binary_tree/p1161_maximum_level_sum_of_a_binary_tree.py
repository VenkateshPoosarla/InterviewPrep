"""1161. Maximum Level Sum of a Binary Tree

Link: https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/

BFS level-by-level. Track the maximum sum; if tie, return smallest level index.
Levels are 1-indexed.
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
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0

        q: Deque[TreeNode] = deque([root])
        best_sum = -10**18
        best_level = 1
        level = 1

        while q:
            level_sum = 0
            for _ in range(len(q)):
                node = q.popleft()
                level_sum += node.val
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)

            if level_sum > best_sum:
                best_sum = level_sum
                best_level = level
            level += 1

        return best_level


def run_tests() -> None:
    sol = Solution()

    root = build_tree_level_order([1, 7, 0, 7, -8, None, None])
    assert sol.maxLevelSum(root) == 2

    root = build_tree_level_order([989, None, 10250, 98693, -89388, None, None, None, -32127])
    assert sol.maxLevelSum(root) == 2


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


