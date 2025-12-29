"""2462. Total Cost to Hire K Workers

Link: https://leetcode.com/problems/total-cost-to-hire-k-workers/

We hire k workers. At each step we can pick the smallest cost from:
- left candidate window (from start)
- right candidate window (from end)
where each window has up to `candidates` workers not yet taken.

Use two min-heaps (left/right) and two pointers.
"""

from __future__ import annotations

import heapq
from typing import List


class Solution:
    def totalCost(self, costs: List[int], k: int, candidates: int) -> int:
        n = len(costs)
        left: list[int] = []
        right: list[int] = []

        i = 0
        j = n - 1

        # Fill initial windows.
        while i <= j and len(left) < candidates:
            heapq.heappush(left, costs[i])
            i += 1
        while j >= i and len(right) < candidates:
            heapq.heappush(right, costs[j])
            j -= 1

        total = 0
        for _ in range(k):
            # pick best among heap tops
            if right and (not left or right[0] < left[0]):
                total += heapq.heappop(right)
                if j >= i:
                    heapq.heappush(right, costs[j])
                    j -= 1
            else:
                total += heapq.heappop(left)
                if i <= j:
                    heapq.heappush(left, costs[i])
                    i += 1
        return total


def run_tests() -> None:
    sol = Solution()

    assert sol.totalCost([17, 12, 10, 2, 7, 2, 11, 20, 8], 3, 4) == 11
    assert sol.totalCost([1, 2, 4, 1], 3, 3) == 4
    assert sol.totalCost([5, 5, 5], 2, 1) == 10


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


