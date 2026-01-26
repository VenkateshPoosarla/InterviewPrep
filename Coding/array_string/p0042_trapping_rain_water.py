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
        n=len(height)
        left_max_wall=[0]*n # max left wall
        right_max_wall=[0]*n # max right wall
        current_max=0 # current max
        #max wall support from left
        for i in range(1,len(height)):
            current_max=max(current_max,height[i-1])
            left_max_wall[i] =current_max
        current_max=0
        #max wall support from right
        for i in range(len(height)-2,-1,-1):
            current_max=max(current_max,height[i+1])
            right_max_wall[i] = current_max
        # water potential storage at each level Min(M_L[i],M_R[i])
        # level = max(h[i]- Min(M_L[i],M_R[i]),0) 
        total_water=0
        for i in range(len(height)):
            total_water+=max(min(left_max_wall[i],right_max_wall[i])-height[i],0) 
        return total_water


def run_tests() -> None:
    sol = Solution()
    assert sol.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
    assert sol.trap([4, 2, 0, 3, 2, 5]) == 9
    assert sol.trap([]) == 0
    assert sol.trap([2, 0, 2]) == 2


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
