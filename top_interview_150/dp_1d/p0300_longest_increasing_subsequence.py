"""300. Longest Increasing Subsequence

Link: https://leetcode.com/problems/longest-increasing-subsequence/

Problem:
Return the length of the longest strictly increasing subsequence.

Approach (patience sorting / tails + binary search):
Maintain `tails[len]` = the minimum possible tail value of an increasing subsequence
of length len+1.
For each x:
- Find the leftmost index i where tails[i] >= x and set tails[i] = x.
If x is bigger than all tails, append it.

Length of tails is the LIS length.

Complexity:
- Time: O(n log n)
- Space: O(n)
"""

from __future__ import annotations

import bisect
import sys
from typing import List


class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        tails: List[int] = []
        for x in nums:
            i = bisect.bisect_left(tails, x)
            if i == len(tails):
                tails.append(x)
            else:
                tails[i] = x
        return len(tails)


def run_tests() -> None:
    sol = Solution()
    assert sol.lengthOfLIS([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert sol.lengthOfLIS([0, 1, 0, 3, 2, 3]) == 4
    assert sol.lengthOfLIS([7, 7, 7, 7, 7]) == 1
    assert sol.lengthOfLIS([]) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
