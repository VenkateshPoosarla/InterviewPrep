"""55. Jump Game

Link: https://leetcode.com/problems/jump-game/

Problem:
Given an array nums where nums[i] is the maximum jump length from index i,
determine if you can reach the last index starting at index 0.

Approach (greedy farthest reach):
Maintain `farthest` = the farthest index reachable so far.
For each i in [0..], if i > farthest we are stuck (can't even reach i).
Otherwise, update farthest = max(farthest, i + nums[i]).
If farthest reaches last index, return True.

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def canJump(self, nums: List[int]) -> bool:
        farthest = 0
        last = len(nums) - 1
        for i, jump in enumerate(nums):
            if i > farthest:
                return False
            farthest = max(farthest, i + jump)
            if farthest >= last:
                return True
        return True


def run_tests() -> None:
    sol = Solution()
    assert sol.canJump([2, 3, 1, 1, 4]) is True
    assert sol.canJump([3, 2, 1, 0, 4]) is False
    assert sol.canJump([0]) is True
    assert sol.canJump([1, 0, 1, 0]) is False


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
