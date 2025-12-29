"""274. H-Index

Link: https://leetcode.com/problems/h-index/

Problem:
Given an array citations where citations[i] is the number of citations for a paper,
return the researcher's h-index.

Definition:
h is the largest value such that there are at least h papers with >= h citations.

Approach (sort descending):
Sort citations in decreasing order. The i-th paper (0-indexed) has citations[i].
We can have h = i+1 if citations[i] >= i+1.
Scan and keep the maximum such i+1.

Complexity:
- Time: O(n log n) (sorting)
- Space: O(1) extra (ignoring sort internals)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def hIndex(self, citations: List[int]) -> int:
        citations.sort(reverse=True)
        h = 0
        for i, c in enumerate(citations):
            if c >= i + 1:
                h = i + 1
            else:
                break
        return h


def run_tests() -> None:
    sol = Solution()
    assert sol.hIndex([3, 0, 6, 1, 5]) == 3
    assert sol.hIndex([1, 3, 1]) == 1
    assert sol.hIndex([0]) == 0
    assert sol.hIndex([100]) == 1


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
