"""2095. Delete the Middle Node of a Linked List

Link: https://leetcode.com/problems/delete-the-middle-node-of-a-linked-list/

Use slow/fast pointers:
- fast moves 2 steps, slow moves 1 step
- keep prev pointer to node before slow
When fast reaches end, slow is at middle.
Delete by prev.next = slow.next.

Edge case:
Single node => result is empty list.
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
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return None

        slow: ListNode = head
        fast: ListNode = head
        prev: Optional[ListNode] = None

        while fast is not None and fast.next is not None:
            prev = slow
            slow = slow.next  # type: ignore[assignment]
            fast = fast.next.next  # type: ignore[assignment]

        # slow is middle; prev exists because len>=2
        prev.next = slow.next  # type: ignore[union-attr]
        return head


def run_tests() -> None:
    sol = Solution()

    head = build_linked_list([1, 3, 4, 7, 1, 2, 6])
    out = sol.deleteMiddle(head)
    assert linked_list_to_list(out) == [1, 3, 4, 1, 2, 6]

    head = build_linked_list([1, 2, 3, 4])
    out = sol.deleteMiddle(head)
    assert linked_list_to_list(out) == [1, 2, 4]

    head = build_linked_list([2, 1])
    out = sol.deleteMiddle(head)
    assert linked_list_to_list(out) == [2]

    head = build_linked_list([1])
    out = sol.deleteMiddle(head)
    assert linked_list_to_list(out) == []


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


