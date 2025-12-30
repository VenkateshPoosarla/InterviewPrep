"""238. Product of Array Except Self

Link: https://leetcode.com/problems/product-of-array-except-self/

Constraints forbid division.

Idea:
Prefix products + suffix products, merged into one output array.

Visual for nums = [a, b, c, d]
  prefix: [1, a, a*b, a*b*c]
  suffix: [b*c*d, c*d, d, 1]
  answer: [1*b*c*d, a*1*c*d, a*b*1*d, a*b*c*1]
"""

from __future__ import annotations

from typing import List


class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        out = [1] * n

        prefix = 1
        for i in range(n):
            out[i] = prefix
            prefix *= nums[i]

        suffix = 1
        for i in range(n - 1, -1, -1):
            out[i] *= suffix
            suffix *= nums[i]

        return out


def run_tests() -> None:
    sol = Solution()

    assert sol.productExceptSelf([1, 2, 3, 4]) == [24, 12, 8, 6]
    assert sol.productExceptSelf([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]
    assert sol.productExceptSelf([0, 0]) == [0, 0]
    assert sol.productExceptSelf([5]) == [1]


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


