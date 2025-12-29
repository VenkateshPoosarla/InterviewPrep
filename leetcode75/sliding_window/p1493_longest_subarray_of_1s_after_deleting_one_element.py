"""1493. Longest Subarray of 1's After Deleting One Element

Link: https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/

We must delete exactly one element. Equivalent:
Find the longest window that contains at most ONE zero,
then answer is window_length - 1 (we "delete" one element, ideally a zero).

Edge case:
All ones => longest window = n, but must delete one => answer = n - 1.
"""

from __future__ import annotations

from typing import List


class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        left = 0
        zeros = 0
        best = 0

        for right, val in enumerate(nums):
            if val == 0:
                zeros += 1
            while zeros > 1:
                if nums[left] == 0:
                    zeros -= 1
                left += 1
            best = max(best, right - left + 1)

        # delete one element
        return max(0, best - 1)


def run_tests() -> None:
    sol = Solution()

    assert sol.longestSubarray([1, 1, 0, 1]) == 3
    assert sol.longestSubarray([0, 1, 1, 1, 0, 1, 1, 0, 1]) == 5
    assert sol.longestSubarray([1, 1, 1]) == 2
    assert sol.longestSubarray([0, 0, 0]) == 0
    assert sol.longestSubarray([]) == 0


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


