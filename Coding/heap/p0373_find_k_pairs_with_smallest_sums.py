"""373. Find K Pairs with Smallest Sums

Link: https://leetcode.com/problems/find-k-pairs-with-smallest-sums/

Problem:
Given two sorted arrays nums1 and nums2, return k pairs (u,v) with the smallest sums.

Approach (min-heap over pair indices):
If nums1 and nums2 are sorted ascending, then for each i, pairs (i,0), (i,1), ... have
increasing sums. We can do a k-way merge:
- Seed heap with (i,0) for i in [0..min(k,len(nums1))-1]
- Pop smallest, push (i, j+1) for the same i

Complexity:
- Time: O(k log min(k, len(nums1)))
- Space: O(min(k, len(nums1)))
"""

from __future__ import annotations

import heapq
import sys
from typing import List, Tuple


class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        if not nums1 or not nums2 or k <= 0:
            return []

        heap: List[Tuple[int, int, int]] = []
        for i in range(min(k, len(nums1))):
            heapq.heappush(heap, (nums1[i] + nums2[0], i, 0))

        res: List[List[int]] = []
        while heap and len(res) < k:
            _, i, j = heapq.heappop(heap)
            res.append([nums1[i], nums2[j]])
            if j + 1 < len(nums2):
                heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
        return res


def run_tests() -> None:
    sol = Solution()
    out = sol.kSmallestPairs([1, 7, 11], [2, 4, 6], 3)
    assert out == [[1, 2], [1, 4], [1, 6]]
    out = sol.kSmallestPairs([1, 1, 2], [1, 2, 3], 2)
    assert out == [[1, 1], [1, 1]]
    out = sol.kSmallestPairs([1, 2], [3], 3)
    assert out == [[1, 3], [2, 3]]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
