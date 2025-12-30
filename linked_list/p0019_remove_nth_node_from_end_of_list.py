"""19. Remove Nth Node From End of List

Link: https://leetcode.com/problems/remove-nth-node-from-end-of-list/

Problem:
Remove the n-th node from the end of the list and return the head.

Approach (two pointers with a dummy head):
Use a dummy before head to simplify removing the first node.
Advance `fast` by n steps, then move `fast` and `slow` together until fast hits the end.
`slow.next` is then the node to remove.

Complexity:
- Time: O(L)
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
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        slow: ListNode = dummy
        fast: Optional[ListNode] = dummy

        for _ in range(n):
            assert fast is not None
            fast = fast.next

        while fast is not None and fast.next is not None:
            fast = fast.next
            slow = slow.next  # type: ignore[assignment]

        # remove slow.next
        if slow.next is not None:
            slow.next = slow.next.next
        return dummy.next


def run_tests() -> None:
    sol = Solution()

    head = build_linked_list([1, 2, 3, 4, 5])
    out = sol.removeNthFromEnd(head, 2)
    assert linked_list_to_list(out) == [1, 2, 3, 5]

    head = build_linked_list([1])
    out = sol.removeNthFromEnd(head, 1)
    assert linked_list_to_list(out) == []

    head = build_linked_list([1, 2])
    out = sol.removeNthFromEnd(head, 1)
    assert linked_list_to_list(out) == [1]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
