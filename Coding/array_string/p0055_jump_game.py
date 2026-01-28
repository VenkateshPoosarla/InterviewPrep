"""55. Jump Game

Link: https://leetcode.com/problems/jump-game/

Problem:
Given an array nums where nums[i] is the maximum jump length from index i,
determine if you can reach the last index starting at index 0.

Approach (greedy farthest reach):
For each position, the goal index represents the leftmost index that can reach the end of the array.
While iterating from right to left, if the current index can jump to or beyond the existing goal index,
 we mark it as reachable and update the goal index to the minimum of the current index and the previous goal index.

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n=len(nums)
        goal=[False]*n
        last_success=n-1
        for i in range(n-1,-1,-1):
            if i+nums[i]>=last_success:
                goal[i]=True
                last_success=min(i,last_success)
        return goal[0]


def run_tests() -> None:
    sol = Solution()
    assert sol.canJump([2, 3, 1, 1, 4]) is True
    assert sol.canJump([3, 2, 1, 0, 4]) is False
    assert sol.canJump([0]) is True
    assert sol.canJump([1, 0, 1, 0]) is False


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
