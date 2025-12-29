"""437. Path Sum III

Link: https://leetcode.com/problems/path-sum-iii/

Count number of paths (downwards) whose sum equals targetSum.

Prefix-sum on paths from root:
- Let prefix = sum along current root->node path.
- A path ending at this node has sum target iff there was a previous prefix = prefix - target.

Maintain a hashmap count[prefix] while DFS.
"""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from typing import Deque, DefaultDict, List, Optional


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
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        counts: DefaultDict[int, int] = defaultdict(int)
        counts[0] = 1

        def dfs(node: Optional[TreeNode], prefix: int) -> int:
            if node is None:
                return 0

            prefix += node.val
            res = counts[prefix - targetSum]

            counts[prefix] += 1
            res += dfs(node.left, prefix)
            res += dfs(node.right, prefix)
            counts[prefix] -= 1

            return res

        return dfs(root, 0)


def run_tests() -> None:
    sol = Solution()

    root = build_tree_level_order([10, 5, -3, 3, 2, None, 11, 3, -2, None, 1])
    assert sol.pathSum(root, 8) == 3

    root = build_tree_level_order([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, 5, 1])
    assert sol.pathSum(root, 22) == 3

    assert sol.pathSum(build_tree_level_order([]), 0) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


