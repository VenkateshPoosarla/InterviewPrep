"""2542. Maximum Subsequence Score

Link: https://leetcode.com/problems/maximum-subsequence-score/

We choose k indices. Score = (sum(nums1[i])) * min(nums2[i]).

Sort pairs by nums2 descending. When we sweep:
- treat current nums2 as the minimum (because remaining are <= current).
- keep the best possible sum(nums1) of k items among seen using a min-heap of nums1.

For each step where we have k items, compute sum_heap * current_nums2 and maximize.
"""

from __future__ import annotations

import heapq
from typing import List, Tuple


class Solution:
    def maxScore(self, nums1: List[int], nums2: List[int], k: int) -> int:
        pairs: List[Tuple[int, int]] = sorted(zip(nums2, nums1), reverse=True)  # (nums2, nums1)

        heap: list[int] = []
        s = 0
        best = 0

        for n2, n1 in pairs:
            heapq.heappush(heap, n1)
            s += n1
            if len(heap) > k:
                s -= heapq.heappop(heap)
            if len(heap) == k:
                best = max(best, s * n2)

        return best


def run_tests() -> None:
    sol = Solution()

    assert sol.maxScore([1, 3, 3, 2], [2, 1, 3, 4], 3) == 12
    assert sol.maxScore([4, 2, 3, 1, 1], [7, 5, 10, 9, 6], 1) == 30


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


