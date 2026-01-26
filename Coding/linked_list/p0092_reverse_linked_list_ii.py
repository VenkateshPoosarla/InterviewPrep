"""92. Reverse Linked List II

Link: https://leetcode.com/problems/reverse-linked-list-ii/

Problem:
Reverse a linked list from position left to position right (1-indexed), in-place.

Approach (head insertion within the sublist):
Use a dummy head. Walk `pre` to the node before `left`.
Then repeatedly take the node after `cur` and insert it right after `pre`,
effectively reversing the sublist in-place.

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
    def reverseBetween(
        self, head: Optional[ListNode], left: int, right: int
    ) -> Optional[ListNode]:
        if head is None or left == right:
            return head

        dummy = ListNode(0, head)
        pre: ListNode = dummy
        for _ in range(left - 1):
            pre = pre.next  # type: ignore[assignment]

        cur = pre.next
        # reverse by moving nodes after `cur` to the front of the sublist
        for _ in range(right - left):
            assert cur is not None and cur.next is not None
            nxt = cur.next
            cur.next = nxt.next
            nxt.next = pre.next
            pre.next = nxt

        return dummy.next


def run_tests() -> None:
    sol = Solution()

    head = build_linked_list([1, 2, 3, 4, 5])
    out = sol.reverseBetween(head, 2, 4)
    assert linked_list_to_list(out) == [1, 4, 3, 2, 5]

    head = build_linked_list([5])
    out = sol.reverseBetween(head, 1, 1)
    assert linked_list_to_list(out) == [5]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
