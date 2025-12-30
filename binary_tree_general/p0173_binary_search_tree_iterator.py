"""173. Binary Search Tree Iterator

Link: https://leetcode.com/problems/binary-search-tree-iterator/

Problem:
Implement an iterator over a BST that returns the next smallest number.
Methods:
- next() -> int
- hasNext() -> bool

Requirement: next/hasNext should be average O(1) time and O(h) space.

Approach (controlled inorder traversal with a stack):
Inorder traversal of BST yields sorted order.
Maintain a stack of nodes representing the path to the current next node.
`_push_left(node)` pushes node and all its left descendants.
- next(): pop top, then push left path of popped.right
- hasNext(): stack is non-empty

Complexity:
- Time: O(1) amortized per next()
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


class BSTIterator:
    def __init__(self, root: Optional[TreeNode]) -> None:
        self.stack: list[TreeNode] = []
        self._push_left(root)

    def _push_left(self, node: Optional[TreeNode]) -> None:
        while node is not None:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        node = self.stack.pop()
        if node.right is not None:
            self._push_left(node.right)
        return node.val

    def hasNext(self) -> bool:
        return bool(self.stack)


def run_tests() -> None:
    root = build_tree_level_order([7, 3, 15, None, None, 9, 20])
    it = BSTIterator(root)
    assert it.next() == 3
    assert it.next() == 7
    assert it.hasNext() is True
    assert it.next() == 9
    assert it.hasNext() is True
    assert it.next() == 15
    assert it.hasNext() is True
    assert it.next() == 20
    assert it.hasNext() is False


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
