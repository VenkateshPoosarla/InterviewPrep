"""215. Kth Largest Element in an Array

Link: https://leetcode.com/problems/kth-largest-element-in-an-array/

Use a min-heap of size k:
- push each number
- if heap grows beyond k, pop smallest
At end, heap[0] is kth largest.
"""

from __future__ import annotations

import heapq
from typing import List


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        h: list[int] = []
        for x in nums:
            heapq.heappush(h, x)
            if len(h) > k:
                heapq.heappop(h)
        return h[0]


def run_tests() -> None:
    sol = Solution()

    assert sol.findKthLargest([3, 2, 1, 5, 6, 4], 2) == 5
    assert sol.findKthLargest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
    assert sol.findKthLargest([1], 1) == 1


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


