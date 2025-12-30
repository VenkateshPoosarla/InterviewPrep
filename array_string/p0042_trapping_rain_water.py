"""42. Trapping Rain Water

Link: https://leetcode.com/problems/trapping-rain-water/

Problem:
Given elevation map `height`, compute how much water it can trap after raining.

Approach (two pointers):
At any index i, trapped water is:
  max(0, min(max_left[i], max_right[i]) - height[i])

Instead of precomputing arrays, use two pointers:
- Track `left_max` and `right_max`
- Move the pointer with the smaller current height inward; that side’s max determines
  how much water can be trapped at that position.

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0

        l, r = 0, len(height) - 1
        left_max = 0
        right_max = 0
        water = 0

        while l < r:
            if height[l] <= height[r]:
                if height[l] >= left_max:
                    left_max = height[l]
                else:
                    water += left_max - height[l]
                l += 1
            else:
                if height[r] >= right_max:
                    right_max = height[r]
                else:
                    water += right_max - height[r]
                r -= 1
        return water


def run_tests() -> None:
    sol = Solution()
    assert sol.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
    assert sol.trap([4, 2, 0, 3, 2, 5]) == 9
    assert sol.trap([]) == 0
    assert sol.trap([2, 0, 2]) == 2


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
