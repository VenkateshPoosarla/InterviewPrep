"""61. Rotate List

Link: https://leetcode.com/problems/rotate-list/

Problem:
Rotate a linked list to the right by k places.

Approach:
1) Compute length n and get the tail.
2) k %= n; if k == 0 return head.
3) Make the list circular: tail.next = head.
4) New tail is at position (n - k - 1) from head; new head is new_tail.next.
5) Break the circle.

Complexity:
- Time: O(n)
- Space: O(1)
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
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if head is None or head.next is None or k == 0:
            return head

        # length + tail
        n = 1
        tail = head
        while tail.next is not None:
            tail = tail.next
            n += 1

        k %= n
        if k == 0:
            return head

        # make circular
        tail.next = head

        # find new tail: n-k-1 steps from head
        steps = n - k - 1
        new_tail = head
        for _ in range(steps):
            new_tail = new_tail.next  # type: ignore[assignment]

        new_head = new_tail.next
        new_tail.next = None
        return new_head


def run_tests() -> None:
    sol = Solution()

    head = build_linked_list([1, 2, 3, 4, 5])
    out = sol.rotateRight(head, 2)
    assert linked_list_to_list(out) == [4, 5, 1, 2, 3]

    head = build_linked_list([0, 1, 2])
    out = sol.rotateRight(head, 4)
    assert linked_list_to_list(out) == [2, 0, 1]

    head = build_linked_list([1])
    out = sol.rotateRight(head, 10)
    assert linked_list_to_list(out) == [1]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
