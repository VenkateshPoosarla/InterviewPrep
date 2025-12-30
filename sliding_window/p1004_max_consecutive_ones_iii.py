"""1004. Max Consecutive Ones III

Link: https://leetcode.com/problems/max-consecutive-ones-iii/

Sliding window (variable size):
We can flip at most k zeros inside the window. Expand right; if zeros > k, shrink left.
Answer is max window length seen.
"""

from __future__ import annotations

from typing import List


class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        left = 0
        zeros = 0
        best = 0

        for right, val in enumerate(nums):
            if val == 0:
                zeros += 1
            while zeros > k:
                if nums[left] == 0:
                    zeros -= 1
                left += 1
            best = max(best, right - left + 1)

        return best


def run_tests() -> None:
    sol = Solution()

    assert sol.longestOnes([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2) == 6
    assert sol.longestOnes([0, 0, 1, 1, 1, 0, 0], 0) == 3
    assert sol.longestOnes([1, 1, 1], 2) == 3
    assert sol.longestOnes([], 3) == 0


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


