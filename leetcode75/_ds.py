"""Shared data-structure helpers for LeetCode-style problems.

Some LeetCode-75 problems use linked lists / binary trees. To avoid repeating
boilerplate, those problem files may import from this module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional


@dataclass
class ListNode:
    val: int
    next: Optional["ListNode"] = None


def build_linked_list(values: Iterable[int]) -> Optional[ListNode]:
    """Build a singly-linked list from Python values."""
    dummy = ListNode(0)
    cur = dummy
    for v in values:
        cur.next = ListNode(int(v))
        cur = cur.next
    return dummy.next


def linked_list_to_list(head: Optional[ListNode]) -> List[int]:
    """Convert a linked list to a Python list."""
    out: List[int] = []
    cur = head
    while cur is not None:
        out.append(cur.val)
        cur = cur.next
    return out


@dataclass
class TreeNode:
    val: int
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


def build_tree_level_order(values: List[Optional[int]]) -> Optional[TreeNode]:
    """Build a binary tree from a LeetCode level-order array (None = missing).

    Example:
        [1, 2, 3, None, 4] corresponds to:

              1
             / \
            2   3
             \
              4
    """
    if not values:
        return None
    if values[0] is None:
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


def tree_to_level_order(root: Optional[TreeNode]) -> List[Optional[int]]:
    """Convert a binary tree to level-order list with trailing Nones trimmed."""
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

    # Trim trailing Nones (LeetCode commonly omits them).
    while out and out[-1] is None:
        out.pop()
    return out


