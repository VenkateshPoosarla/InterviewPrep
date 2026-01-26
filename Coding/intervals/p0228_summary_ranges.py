"""228. Summary Ranges

Link: https://leetcode.com/problems/summary-ranges/

Problem:
Given a sorted array of unique integers, return the smallest list of ranges that cover
all numbers exactly.

Example:
  [0,1,2,4,5,7] -> ["0->2","4->5","7"]

Approach:
Walk through nums, tracking the start of the current run. When the run breaks, emit
either "a" or "a->b".

Complexity:
- Time: O(n)
- Space: O(1) extra (output excluded)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def summaryRanges(self, nums: List[int]) -> List[str]:
        if not nums:
            return []

        res: List[str] = []
        start = nums[0]
        prev = nums[0]

        for x in nums[1:]:
            if x == prev + 1:
                prev = x
                continue
            # end current range
            if start == prev:
                res.append(f"{start}")
            else:
                res.append(f"{start}->{prev}")
            start = prev = x

        # finalize last range
        if start == prev:
            res.append(f"{start}")
        else:
            res.append(f"{start}->{prev}")
        return res


def run_tests() -> None:
    sol = Solution()
    assert sol.summaryRanges([0, 1, 2, 4, 5, 7]) == ["0->2", "4->5", "7"]
    assert sol.summaryRanges([0, 2, 3, 4, 6, 8, 9]) == ["0", "2->4", "6", "8->9"]
    assert sol.summaryRanges([]) == []
    assert sol.summaryRanges([1]) == ["1"]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
