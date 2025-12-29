"""86. Partition List

Link: https://leetcode.com/problems/partition-list/

Problem:
Given the head of a linked list and a value x, partition it so that all nodes with
val < x come before nodes with val >= x, preserving original relative order within
each partition.

Approach (two lists):
Build two chains:
- `small` for nodes < x
- `large` for nodes >= x
Then connect small tail to large head.

Complexity:
- Time: O(n)
- Space: O(1) extra (re-links nodes)
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
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        small_dummy = ListNode(0)
        large_dummy = ListNode(0)
        small = small_dummy
        large = large_dummy

        cur = head
        while cur is not None:
            nxt = cur.next
            cur.next = None
            if cur.val < x:
                small.next = cur
                small = small.next
            else:
                large.next = cur
                large = large.next
            cur = nxt

        small.next = large_dummy.next
        return small_dummy.next


def run_tests() -> None:
    sol = Solution()

    head = build_linked_list([1, 4, 3, 2, 5, 2])
    out = sol.partition(head, 3)
    assert linked_list_to_list(out) == [1, 2, 2, 4, 3, 5]

    head = build_linked_list([2, 1])
    out = sol.partition(head, 2)
    assert linked_list_to_list(out) == [1, 2]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
