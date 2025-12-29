"""643. Maximum Average Subarray I

Link: https://leetcode.com/problems/maximum-average-subarray-i/

Sliding window of fixed length k.
Maintain running sum of current window in O(1) per step.
"""

from __future__ import annotations

from typing import List


class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        window_sum = sum(nums[:k])
        best = window_sum
        for i in range(k, len(nums)):
            window_sum += nums[i] - nums[i - k]
            if window_sum > best:
                best = window_sum
        return best / k


def run_tests() -> None:
    sol = Solution()

    assert abs(sol.findMaxAverage([1, 12, -5, -6, 50, 3], 4) - 12.75) < 1e-9
    assert abs(sol.findMaxAverage([5], 1) - 5.0) < 1e-9
    assert abs(sol.findMaxAverage([-1, -2, -3], 2) - (-1.5)) < 1e-9


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


