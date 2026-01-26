"""25. Reverse Nodes in k-Group

Link: https://leetcode.com/problems/reverse-nodes-in-k-group/

Problem:
Given a linked list, reverse the nodes of a list k at a time and return its modified list.
Nodes that do not form a complete group of k at the end remain as-is.

Approach (iterate groups + reverse pointers):
Use a dummy head and process groups:
1) Find the k-th node from `group_prev`. If fewer than k nodes remain, stop.
2) Reverse nodes in [group_prev.next .. kth] by pointer rewiring.
3) Reconnect the reversed group and advance `group_prev` to the end of the group.

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
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if head is None or k <= 1:
            return head

        dummy = ListNode(0, head)
        group_prev: ListNode = dummy

        def get_kth(start: ListNode, k: int) -> Optional[ListNode]:
            cur: Optional[ListNode] = start
            for _ in range(k):
                if cur is None:
                    return None
                cur = cur.next
            return cur

        while True:
            kth = get_kth(group_prev, k)
            if kth is None:
                break
            group_next = kth.next

            # reverse group
            prev = group_next
            cur = group_prev.next
            while cur is not group_next:
                nxt = cur.next  # type: ignore[union-attr]
                cur.next = prev  # type: ignore[union-attr]
                prev = cur
                cur = nxt

            # connect
            old_group_start = group_prev.next
            group_prev.next = kth
            group_prev = old_group_start  # type: ignore[assignment]

        return dummy.next


def run_tests() -> None:
    sol = Solution()

    head = build_linked_list([1, 2, 3, 4, 5])
    out = sol.reverseKGroup(head, 2)
    assert linked_list_to_list(out) == [2, 1, 4, 3, 5]

    head = build_linked_list([1, 2, 3, 4, 5])
    out = sol.reverseKGroup(head, 3)
    assert linked_list_to_list(out) == [3, 2, 1, 4, 5]

    head = build_linked_list([1])
    out = sol.reverseKGroup(head, 1)
    assert linked_list_to_list(out) == [1]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
