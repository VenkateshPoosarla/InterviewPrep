"""283. Move Zeroes

Link: https://leetcode.com/problems/move-zeroes/

Requirement: modify nums in-place, keep relative order of non-zeros.

Two-pointer (write index):
- write points to where next non-zero should go.
- scan i from left to right; when nums[i] != 0, swap into write.

Visual:
    nums: 0 1 0 3 12
    write=0
    i=1 -> swap nums[1] with nums[0] => 1 0 0 3 12  (write=1)
    i=3 -> swap nums[3] with nums[1] => 1 3 0 0 12  (write=2)
    i=4 -> swap nums[4] with nums[2] => 1 3 12 0 0  (write=3)
"""

from __future__ import annotations

from typing import List


class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        write = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[write], nums[i] = nums[i], nums[write]
                write += 1


def run_tests() -> None:
    sol = Solution()

    nums = [0, 1, 0, 3, 12]
    sol.moveZeroes(nums)
    assert nums == [1, 3, 12, 0, 0]

    nums = [0]
    sol.moveZeroes(nums)
    assert nums == [0]

    nums = [1, 2, 3]
    sol.moveZeroes(nums)
    assert nums == [1, 2, 3]

    nums = [0, 0, 0, 1]
    sol.moveZeroes(nums)
    assert nums == [1, 0, 0, 0]


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


