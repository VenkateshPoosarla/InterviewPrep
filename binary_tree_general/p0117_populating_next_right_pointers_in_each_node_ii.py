"""117. Populating Next Right Pointers in Each Node II

Link: https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/

Problem:
Given a binary tree, populate each next pointer to point to its next right node.
If there is no next right node, next should be set to None.
Unlike problem 116, the tree is not necessarily perfect.

Approach (level traversal using already-built next pointers, O(1) extra space):
We iterate level by level:
- `cur` walks the current level using next pointers.
- Build the next level using a dummy head and `tail` pointer.
For each node on current level, append its children to the next-level chain.
Then move to dummy.next as the start of next level.

Complexity:
- Time: O(n)
- Space: O(1) extra
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class Node:
    val: int
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    next: Optional["Node"] = None


class Solution:
    def connect(self, root: Optional[Node]) -> Optional[Node]:
        if root is None:
            return None

        level = root
        while level is not None:
            dummy = Node(0)
            tail = dummy
            cur = level
            while cur is not None:
                if cur.left is not None:
                    tail.next = cur.left
                    tail = tail.next
                if cur.right is not None:
                    tail.next = cur.right
                    tail = tail.next
                cur = cur.next
            level = dummy.next
        return root


def run_tests() -> None:
    sol = Solution()
    # Tree:
    #     1
    #   /   \
    #  2     3
    # / \     \
    #4   5     7
    root = Node(1)
    root.left = Node(2, left=Node(4), right=Node(5))
    root.right = Node(3, right=Node(7))

    sol.connect(root)
    assert root.next is None
    assert root.left.next is root.right
    assert root.right.next is None
    assert root.left.left.next is root.left.right
    assert root.left.right.next is root.right.right
    assert root.right.right.next is None


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
