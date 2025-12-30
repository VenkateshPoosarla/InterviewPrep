"""2. Add Two Numbers

Link: https://leetcode.com/problems/add-two-numbers/

Problem:
Two non-empty linked lists represent two non-negative integers in reverse order.
Each node contains a single digit. Add the two numbers and return the sum as a linked list.

Approach:
Simulate grade-school addition with a carry:
- sum = digit1 + digit2 + carry
- new_digit = sum % 10, carry = sum // 10

Complexity:
- Time: O(max(m, n))
- Space: O(max(m, n)) for output list
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
    def addTwoNumbers(
        self, l1: Optional[ListNode], l2: Optional[ListNode]
    ) -> Optional[ListNode]:
        dummy = ListNode(0)
        tail = dummy
        carry = 0

        a, b = l1, l2
        while a is not None or b is not None or carry:
            x = a.val if a is not None else 0
            y = b.val if b is not None else 0
            s = x + y + carry
            carry, digit = divmod(s, 10)
            tail.next = ListNode(digit)
            tail = tail.next
            a = a.next if a is not None else None
            b = b.next if b is not None else None

        return dummy.next


def run_tests() -> None:
    sol = Solution()

    out = sol.addTwoNumbers(build_linked_list([2, 4, 3]), build_linked_list([5, 6, 4]))
    assert linked_list_to_list(out) == [7, 0, 8]

    out = sol.addTwoNumbers(build_linked_list([0]), build_linked_list([0]))
    assert linked_list_to_list(out) == [0]

    out = sol.addTwoNumbers(build_linked_list([9, 9, 9, 9, 9, 9, 9]), build_linked_list([9, 9, 9, 9]))
    assert linked_list_to_list(out) == [8, 9, 9, 9, 0, 0, 0, 1]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
