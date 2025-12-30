"""1679. Max Number of K-Sum Pairs

Link: https://leetcode.com/problems/max-number-of-k-sum-pairs/

We want the maximum number of disjoint pairs (i, j) with nums[i] + nums[j] == k.

Approach (sort + two pointers):
- sort nums
- l at start, r at end
  - if nums[l] + nums[r] == k: count++, l++, r--
  - if sum < k: l++
  - if sum > k: r--

Visual (k=5):
  nums sorted: 1 1 2 3 4
               l     r  => 1+4=5 pair
                 l r    => 1+3=4 <5 => l++
                   lr   => 2+3=5 pair
"""

from __future__ import annotations

from typing import List


class Solution:
    def maxOperations(self, nums: List[int], k: int) -> int:
        nums = sorted(nums)
        l, r = 0, len(nums) - 1
        ops = 0
        while l < r:
            s = nums[l] + nums[r]
            if s == k:
                ops += 1
                l += 1
                r -= 1
            elif s < k:
                l += 1
            else:
                r -= 1
        return ops


def run_tests() -> None:
    sol = Solution()

    assert sol.maxOperations([1, 2, 3, 4], 5) == 2
    assert sol.maxOperations([3, 1, 3, 4, 3], 6) == 1
    assert sol.maxOperations([], 5) == 0
    assert sol.maxOperations([2, 2, 2, 2], 4) == 2
    assert sol.maxOperations([1, 1, 1], 2) == 1


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


