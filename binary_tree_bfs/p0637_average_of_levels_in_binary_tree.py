"""637. Average of Levels in Binary Tree

Link: https://leetcode.com/problems/average-of-levels-in-binary-tree/

Problem:
Return the average value of the nodes on each level in the form of a list.

Approach (BFS):
For each level, compute sum and count, then append sum / count.

Complexity:
- Time: O(n)
- Space: O(w)
"""

from __future__ import annotations

import sys
from collections import deque
from typing import Deque, List, Optional

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
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        if root is None:
            return []
        res: List[float] = []
        q: Deque[TreeNode] = deque([root])
        while q:
            total = 0
            count = len(q)
            for _ in range(count):
                node = q.popleft()
                total += node.val
                if node.left is not None:
                    q.append(node.left)
                if node.right is not None:
                    q.append(node.right)
            res.append(total / count)
        return res


def run_tests() -> None:
    sol = Solution()
    root = build_tree_level_order([3, 9, 20, None, None, 15, 7])
    assert sol.averageOfLevels(root) == [3.0, 14.5, 11.0]
    assert sol.averageOfLevels(None) == []


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
