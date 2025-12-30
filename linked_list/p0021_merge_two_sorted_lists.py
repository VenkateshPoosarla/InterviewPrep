"""21. Merge Two Sorted Lists

Link: https://leetcode.com/problems/merge-two-sorted-lists/

Problem:
Merge two sorted linked lists and return it as a single sorted list.

Approach (iterative with dummy head):
Walk both lists:
- Take the smaller head node, append it to the result, advance that list.
- When one list ends, append the rest of the other list.

Complexity:
- Time: O(m + n)
- Space: O(1) (re-links existing nodes)
"""

from __future__ import annotations

import sys
from typing import Optional

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
    def mergeTwoLists(
        self, list1: Optional[ListNode], list2: Optional[ListNode]
    ) -> Optional[ListNode]:
        dummy = ListNode(0)
        tail = dummy

        a, b = list1, list2
        while a is not None and b is not None:
            if a.val <= b.val:
                tail.next = a
                a = a.next
            else:
                tail.next = b
                b = b.next
            tail = tail.next

        tail.next = a if a is not None else b
        return dummy.next


def run_tests() -> None:
    sol = Solution()

    l1 = build_linked_list([1, 2, 4])
    l2 = build_linked_list([1, 3, 4])
    out = sol.mergeTwoLists(l1, l2)
    assert linked_list_to_list(out) == [1, 1, 2, 3, 4, 4]

    out = sol.mergeTwoLists(build_linked_list([]), build_linked_list([]))
    assert linked_list_to_list(out) == []

    out = sol.mergeTwoLists(build_linked_list([]), build_linked_list([0]))
    assert linked_list_to_list(out) == [0]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
