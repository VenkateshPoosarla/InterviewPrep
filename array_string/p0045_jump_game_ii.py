"""45. Jump Game II

Link: https://leetcode.com/problems/jump-game-ii/

Problem:
Given nums where nums[i] is the maximum jump length from index i, return the minimum
number of jumps to reach the last index. You can assume you can always reach it.

Approach (greedy "level by level"):
Think of reachable indices as BFS levels on the array:
- `current_end` is the farthest index we can reach with the current number of jumps.
- While scanning indices within the current level, track `farthest` reachable for
  the next jump.
- When we reach `current_end`, we must take a jump, and the next level ends at `farthest`.

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def jump(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return 0

        jumps = 0
        current_end = 0
        farthest = 0

        # We never need to "jump from" the last index.
        for i in range(len(nums) - 1):
            farthest = max(farthest, i + nums[i])
            if i == current_end:
                jumps += 1
                current_end = farthest
        return jumps


def run_tests() -> None:
    sol = Solution()
    assert sol.jump([2, 3, 1, 1, 4]) == 2
    assert sol.jump([2, 3, 0, 1, 4]) == 2
    assert sol.jump([0]) == 0
    assert sol.jump([1, 1, 1, 1]) == 3


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
