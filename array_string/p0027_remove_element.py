"""27. Remove Element

Link: https://leetcode.com/problems/remove-element/

Problem:
Remove all occurrences of `val` in-place from `nums` and return the new length `k`.
The first `k` elements of `nums` should hold the kept values (order can change).

Approach (write index):
Walk through nums and copy each kept element forward into the next write position.

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        write = 0
        for x in nums:
            if x != val:
                nums[write] = x
                write += 1
        return write


def run_tests() -> None:
    sol = Solution()

    nums = [3, 2, 2, 3]
    k = sol.removeElement(nums, 3)
    assert k == 2
    assert sorted(nums[:k]) == [2, 2]

    nums = [0, 1, 2, 2, 3, 0, 4, 2]
    k = sol.removeElement(nums, 2)
    assert k == 5
    assert sorted(nums[:k]) == [0, 0, 1, 3, 4]

    nums = []
    k = sol.removeElement(nums, 1)
    assert k == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
