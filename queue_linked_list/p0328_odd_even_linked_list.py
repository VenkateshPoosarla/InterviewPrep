"""328. Odd Even Linked List

Link: https://leetcode.com/problems/odd-even-linked-list/

Rearrange nodes so that all nodes in odd positions come first, followed by even positions.
Do it in-place using two pointers.

Visual (positions):
  1 -> 2 -> 3 -> 4 -> 5
  odd:  1 -> 3 -> 5
  even: 2 -> 4
  result: 1 -> 3 -> 5 -> 2 -> 4
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
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head

        odd = head
        even = head.next
        even_head = even

        while even is not None and even.next is not None:
            odd.next = even.next
            odd = odd.next

            even.next = odd.next
            even = even.next

        odd.next = even_head
        return head


def run_tests() -> None:
    sol = Solution()

    head = build_linked_list([1, 2, 3, 4, 5])
    out = sol.oddEvenList(head)
    assert linked_list_to_list(out) == [1, 3, 5, 2, 4]

    head = build_linked_list([2, 1, 3, 5, 6, 4, 7])
    out = sol.oddEvenList(head)
    assert linked_list_to_list(out) == [2, 3, 6, 7, 1, 5, 4]

    head = build_linked_list([])
    out = sol.oddEvenList(head)
    assert linked_list_to_list(out) == []


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


