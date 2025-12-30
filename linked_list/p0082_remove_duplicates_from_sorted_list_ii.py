"""82. Remove Duplicates from Sorted List II

Link: https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/

Problem:
Given the head of a sorted linked list, delete all nodes that have duplicate numbers,
leaving only distinct numbers from the original list.

Approach (dummy + skip runs):
Use a dummy head to handle removing the original head.
Walk with `cur`:
- If cur.next and cur.next.next share the same value, this value is duplicated.
  Skip all nodes with that value.
- Else, move forward.

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
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        cur = dummy

        while cur.next is not None:
            if cur.next.next is not None and cur.next.val == cur.next.next.val:
                dup = cur.next.val
                while cur.next is not None and cur.next.val == dup:
                    cur.next = cur.next.next
            else:
                cur = cur.next
        return dummy.next


def run_tests() -> None:
    sol = Solution()

    out = sol.deleteDuplicates(build_linked_list([1, 2, 3, 3, 4, 4, 5]))
    assert linked_list_to_list(out) == [1, 2, 5]

    out = sol.deleteDuplicates(build_linked_list([1, 1, 1, 2, 3]))
    assert linked_list_to_list(out) == [2, 3]

    out = sol.deleteDuplicates(build_linked_list([]))
    assert linked_list_to_list(out) == []


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
