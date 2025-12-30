"""148. Sort List

Link: https://leetcode.com/problems/sort-list/

Problem:
Sort a linked list in ascending order.

Approach (merge sort on linked list):
Use slow/fast pointers to split list into halves, recursively sort, then merge two sorted lists.

Complexity:
- Time: O(n log n)
- Space: O(log n) recursion stack
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
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head

        # split list
        prev: Optional[ListNode] = None
        slow = head
        fast = head
        while fast is not None and fast.next is not None:
            prev = slow
            slow = slow.next  # type: ignore[assignment]
            fast = fast.next.next
        assert prev is not None
        prev.next = None

        left = self.sortList(head)
        right = self.sortList(slow)
        return self._merge(left, right)

    def _merge(self, a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        tail = dummy
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
    out = sol.sortList(build_linked_list([4, 2, 1, 3]))
    assert linked_list_to_list(out) == [1, 2, 3, 4]
    out = sol.sortList(build_linked_list([-1, 5, 3, 4, 0]))
    assert linked_list_to_list(out) == [-1, 0, 3, 4, 5]
    out = sol.sortList(build_linked_list([]))
    assert linked_list_to_list(out) == []


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
