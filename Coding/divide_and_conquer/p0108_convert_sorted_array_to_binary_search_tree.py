"""108. Convert Sorted Array to Binary Search Tree

Link: https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/

Problem:
Given an integer array sorted in ascending order, convert it to a height-balanced BST.

Approach (divide and conquer):
Pick the middle element as root, recursively build left subtree from left half and
right subtree from right half.

Complexity:
- Time: O(n)
- Space: O(log n) recursion stack
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
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def build(lo: int, hi: int) -> Optional[TreeNode]:
            if lo > hi:
                return None
            mid = (lo + hi) // 2
            root = TreeNode(nums[mid])
            root.left = build(lo, mid - 1)
            root.right = build(mid + 1, hi)
            return root

        return build(0, len(nums) - 1)


def run_tests() -> None:
    sol = Solution()
    root = sol.sortedArrayToBST([-10, -3, 0, 5, 9])
    # multiple valid balanced trees; just check it has correct inorder via level order sanity
    assert tree_to_level_order(root)[0] in {0, -3, 5}
    assert sol.sortedArrayToBST([]) is None


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
