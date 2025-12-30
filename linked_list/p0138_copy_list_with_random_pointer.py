"""138. Copy List with Random Pointer

Link: https://leetcode.com/problems/copy-list-with-random-pointer/

Problem:
Each node has a `next` pointer and a `random` pointer that can point to any node or None.
Return a deep copy of the list.

Approach (3-pass interleaving, O(1) extra space):
1) For each original node A, create A' and insert it right after A:
     A -> A' -> B -> B' -> ...
2) Set random pointers on the copies:
     A'.random = A.random.next   (because A.random's copy sits right after it)
3) Detach the interleaved list into original and copied lists.

Complexity:
- Time: O(n)
- Space: O(1) extra (not counting output nodes)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class Node:
    val: int
    next: Optional["Node"] = None
    random: Optional["Node"] = None


class Solution:
    def copyRandomList(self, head: Optional[Node]) -> Optional[Node]:
        if head is None:
            return None

        # 1) interleave copies
        cur = head
        while cur is not None:
            copy = Node(cur.val, cur.next, None)
            cur.next = copy
            cur = copy.next

        # 2) set random pointers on copies
        cur = head
        while cur is not None:
            copy = cur.next
            if cur.random is not None:
                copy.random = cur.random.next  # type: ignore[union-attr]
            cur = copy.next  # type: ignore[union-attr]

        # 3) detach
        dummy = Node(0)
        copy_tail = dummy
        cur = head
        while cur is not None:
            copy = cur.next
            nxt = copy.next  # type: ignore[union-attr]

            copy_tail.next = copy
            copy_tail = copy  # type: ignore[assignment]

            cur.next = nxt
            cur = nxt

        copy_tail.next = None
        return dummy.next


def _build_random_list(values: list[tuple[int, Optional[int]]]) -> Optional[Node]:
    """
    values: [(val, random_index)] where random_index refers to node index in the list.
    """
    nodes = [Node(v) for v, _ in values]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    for i, (_, ri) in enumerate(values):
        nodes[i].random = nodes[ri] if ri is not None else None
    return nodes[0] if nodes else None


def _serialize(head: Optional[Node]) -> list[tuple[int, Optional[int]]]:
    nodes = []
    idx = {}
    cur = head
    while cur is not None:
        idx[cur] = len(nodes)
        nodes.append(cur)
        cur = cur.next
    out: list[tuple[int, Optional[int]]] = []
    for n in nodes:
        out.append((n.val, idx.get(n.random) if n.random is not None else None))
    return out


def run_tests() -> None:
    sol = Solution()

    head = _build_random_list([(7, None), (13, 0), (11, 4), (10, 2), (1, 0)])
    copy = sol.copyRandomList(head)
    assert _serialize(copy) == [(7, None), (13, 0), (11, 4), (10, 2), (1, 0)]
    # Ensure deep copy: nodes should not be the same objects.
    assert copy is not head

    head = _build_random_list([])
    assert sol.copyRandomList(head) is None

    head = _build_random_list([(1, 0)])
    copy = sol.copyRandomList(head)
    assert _serialize(copy) == [(1, 0)]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
