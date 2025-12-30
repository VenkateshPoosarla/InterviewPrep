"""141. Linked List Cycle

Link: https://leetcode.com/problems/linked-list-cycle/

Problem:
Given head of a linked list, determine if the list has a cycle.

Approach (Floyd's Tortoise & Hare):
Use two pointers:
- slow moves 1 step
- fast moves 2 steps
If there's a cycle, fast will eventually meet slow. If fast reaches None, no cycle.

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import Optional

from typing import Iterable, Optional


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


class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = head
        fast = head
        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return True
        return False


def run_tests() -> None:
    sol = Solution()

    # no cycle
    head = build_linked_list([3, 2, 0, -4])
    assert sol.hasCycle(head) is False

    # create a cycle: 3 -> 2 -> 0 -> -4 -> (back to 2)
    head = build_linked_list([3, 2, 0, -4])
    assert head is not None and head.next is not None
    entry = head.next
    tail = head
    while tail.next is not None:
        tail = tail.next
    tail.next = entry
    assert sol.hasCycle(head) is True

    head = build_linked_list([1])
    assert sol.hasCycle(head) is False


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
