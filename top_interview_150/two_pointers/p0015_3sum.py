"""15. 3Sum

Link: https://leetcode.com/problems/3sum/

Problem:
Given an integer array, return all unique triplets [a,b,c] such that a+b+c == 0.
Triplets must be unique (no duplicates in the output).

Approach (sort + two pointers):
Sort nums.
Fix an index i and find pairs (l,r) with nums[l] + nums[r] == -nums[i] using two pointers.
Skip duplicates for i, and skip duplicates when moving l/r after a match.

Complexity:
- Time: O(n^2)
- Space: O(1) extra (excluding output)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res: List[List[int]] = []
        n = len(nums)

        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            # If the smallest possible sum is already > 0, we can stop.
            if nums[i] > 0:
                break

            target = -nums[i]
            l, r = i + 1, n - 1
            while l < r:
                s = nums[l] + nums[r]
                if s == target:
                    res.append([nums[i], nums[l], nums[r]])
                    l += 1
                    r -= 1
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1
                    while l < r and nums[r] == nums[r + 1]:
                        r -= 1
                elif s < target:
                    l += 1
                else:
                    r -= 1

        return res


def run_tests() -> None:
    sol = Solution()

    out = sol.threeSum([-1, 0, 1, 2, -1, -4])
    assert {tuple(t) for t in out} == {(-1, -1, 2), (-1, 0, 1)}

    assert sol.threeSum([0, 1, 1]) == []
    assert sol.threeSum([0, 0, 0]) == [[0, 0, 0]]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
