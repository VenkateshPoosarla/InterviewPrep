"""23. Merge k Sorted Lists

Link: https://leetcode.com/problems/merge-k-sorted-lists/

Problem:
Merge k sorted linked lists into one sorted linked list and return its head.

Approach (min-heap):
Push the head of each non-empty list into a min-heap keyed by node value.
Repeatedly pop the smallest node, append to output, and push its next node if present.

Complexity:
- Time: O(N log k) where N is total nodes across all lists
- Space: O(k) for heap
"""

from __future__ import annotations

import heapq
import sys
from typing import List, Optional, Tuple

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
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        heap: List[Tuple[int, int, ListNode]] = []
        for i, node in enumerate(lists):
            if node is not None:
                heapq.heappush(heap, (node.val, i, node))

        dummy = ListNode(0)
        tail = dummy

        while heap:
            _, i, node = heapq.heappop(heap)
            tail.next = node
            tail = tail.next
            if node.next is not None:
                heapq.heappush(heap, (node.next.val, i, node.next))

        tail.next = None
        return dummy.next


def run_tests() -> None:
    sol = Solution()
    lists = [build_linked_list([1, 4, 5]), build_linked_list([1, 3, 4]), build_linked_list([2, 6])]
    out = sol.mergeKLists(lists)
    assert linked_list_to_list(out) == [1, 1, 2, 3, 4, 4, 5, 6]

    assert linked_list_to_list(sol.mergeKLists([])) == []
    assert linked_list_to_list(sol.mergeKLists([None])) == []


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
