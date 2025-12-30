"""1372. Longest ZigZag Path in a Binary Tree

Link: https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/

ZigZag alternates left/right directions at each step.

DFS returning two values for each node:
- go_left: longest zigzag starting at this node if the next move is left
- go_right: longest zigzag starting at this node if the next move is right

Transitions:
go_left  = 1 + child_left.go_right   (if left child exists else 0)
go_right = 1 + child_right.go_left  (if right child exists else 0)
Answer is max of all go_left/go_right.
"""

from __future__ import annotations

import sys
from collections import deque
from typing import Deque, List, Optional, Tuple


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
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        best = 0

        def dfs(node: Optional[TreeNode]) -> Tuple[int, int]:
            nonlocal best
            if node is None:
                return (0, 0)

            left = dfs(node.left)
            right = dfs(node.right)

            go_left = 1 + left[1] if node.left else 0
            go_right = 1 + right[0] if node.right else 0

            best = max(best, go_left, go_right)
            return (go_left, go_right)

        dfs(root)
        return best


def run_tests() -> None:
    sol = Solution()

    root = build_tree_level_order([1, None, 1, 1, 1, None, None, 1, 1, None, 1])
    assert sol.longestZigZag(root) == 3

    root = build_tree_level_order([1, 1, 1, None, 1, None, None, 1, 1, None, 1])
    assert sol.longestZigZag(root) >= 2

    assert sol.longestZigZag(build_tree_level_order([1])) == 0
    assert sol.longestZigZag(build_tree_level_order([])) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


