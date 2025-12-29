"""918. Maximum Sum Circular Subarray

Link: https://leetcode.com/problems/maximum-sum-circular-subarray/

Problem:
Find the maximum possible sum of a non-empty subarray of `nums`, where the array is circular.

Key idea:
The best circular subarray is either:
- a normal (non-wrapping) max subarray (Kadane), or
- total_sum - (minimum subarray sum)  (i.e., take everything except a "bad" middle chunk)

Edge case:
If all numbers are negative, total_sum - min_subarray would become 0 (invalid because
subarray must be non-empty). In that case, return max_subarray.

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        total = 0
        cur_max = cur_min = 0
        best_max = -10**18
        best_min = 10**18

        for x in nums:
            total += x
            cur_max = max(x, cur_max + x)
            best_max = max(best_max, cur_max)

            cur_min = min(x, cur_min + x)
            best_min = min(best_min, cur_min)

        # all negative -> best_max is the answer
        if best_max < 0:
            return int(best_max)
        return int(max(best_max, total - best_min))


def run_tests() -> None:
    sol = Solution()
    assert sol.maxSubarraySumCircular([1, -2, 3, -2]) == 3
    assert sol.maxSubarraySumCircular([5, -3, 5]) == 10
    assert sol.maxSubarraySumCircular([-3, -2, -3]) == -2


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
