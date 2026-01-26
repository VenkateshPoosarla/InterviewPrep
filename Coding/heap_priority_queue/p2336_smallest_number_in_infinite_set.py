"""2336. Smallest Number in Infinite Set

Link: https://leetcode.com/problems/smallest-number-in-infinite-set/

We conceptually have {1,2,3,...}.

Maintain:
- current pointer `cur` for the smallest number never popped yet.
- a min-heap `added_back` for numbers that were popped earlier and added back.
- a set `in_heap` to avoid duplicates in the heap.

popSmallest():
  if heap non-empty => pop from heap
  else => return cur and increment cur

addBack(num):
  only matters if num < cur and not already in heap.
"""

from __future__ import annotations

import heapq


class SmallestInfiniteSet:
    def __init__(self) -> None:
        self.cur = 1
        self.added_back: list[int] = []
        self.in_heap: set[int] = set()

    def popSmallest(self) -> int:
        if self.added_back:
            x = heapq.heappop(self.added_back)
            self.in_heap.remove(x)
            return x
        x = self.cur
        self.cur += 1
        return x

    def addBack(self, num: int) -> None:
        if num < self.cur and num not in self.in_heap:
            heapq.heappush(self.added_back, num)
            self.in_heap.add(num)


def run_tests() -> None:
    s = SmallestInfiniteSet()
    assert s.popSmallest() == 1
    assert s.popSmallest() == 2
    s.addBack(1)
    assert s.popSmallest() == 1
    assert s.popSmallest() == 3
    assert s.popSmallest() == 4
    s.addBack(2)
    assert s.popSmallest() == 2


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


