"""206. Reverse Linked List

Link: https://leetcode.com/problems/reverse-linked-list/

Iterative:
prev <- None
cur  <- head
while cur:
  nxt = cur.next
  cur.next = prev
  prev = cur
  cur = nxt
return prev
"""

from __future__ import annotations

import sys
from typing import Iterable, List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = int(val)
        self.next = next


def build_linked_list(values: Iterable[int]) -> Optional[ListNode]:
    dummy = ListNode(0)
    cur = dummy
    for v in values:
        cur.next = ListNode(int(v))
        cur = cur.next
    return dummy.next


def linked_list_to_list(head: Optional[ListNode]) -> List[int]:
    out: List[int] = []
    cur = head
    while cur is not None:
        out.append(cur.val)
        cur = cur.next
    return out


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev: Optional[ListNode] = None
        cur = head
        while cur is not None:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        return prev


def run_tests() -> None:
    sol = Solution()

    head = build_linked_list([1, 2, 3, 4, 5])
    out = sol.reverseList(head)
    assert linked_list_to_list(out) == [5, 4, 3, 2, 1]

    head = build_linked_list([1])
    out = sol.reverseList(head)
    assert linked_list_to_list(out) == [1]

    head = build_linked_list([])
    out = sol.reverseList(head)
    assert linked_list_to_list(out) == []


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


