"""2130. Maximum Twin Sum of a Linked List

Link: https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/

Twin sum pairs i-th from start with i-th from end.

Approach:
1) Find middle with slow/fast.
2) Reverse second half.
3) Walk both halves together computing max(left.val + right.val).

Time: O(n)
Space: O(1) extra (in-place reverse).
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
    def pairSum(self, head: Optional[ListNode]) -> int:
        if head is None:
            return 0

        # Find middle (slow ends at start of 2nd half for even length)
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next  # type: ignore[assignment]
            fast = fast.next.next

        # Reverse second half starting from slow
        prev: Optional[ListNode] = None
        cur = slow
        while cur:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt

        # prev is head of reversed second half
        best = 0
        left = head
        right = prev
        while right is not None:
            best = max(best, left.val + right.val)
            left = left.next  # type: ignore[assignment]
            right = right.next
        return best


def run_tests() -> None:
    sol = Solution()

    head = build_linked_list([5, 4, 2, 1])
    assert sol.pairSum(head) == 6

    head = build_linked_list([4, 2, 2, 3])
    assert sol.pairSum(head) == 7

    head = build_linked_list([1, 100000])
    assert sol.pairSum(head) == 100001

    # sanity: no mutation expectations for tests, but ensure list is still traversable
    head = build_linked_list([1, 2, 3, 4])
    _ = sol.pairSum(head)
    assert linked_list_to_list(head)  # at least doesn't crash to traverse


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


