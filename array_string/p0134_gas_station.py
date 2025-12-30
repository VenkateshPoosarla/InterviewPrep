"""134. Gas Station

Link: https://leetcode.com/problems/gas-station/

Problem:
There are n gas stations in a circle. At station i you gain gas[i] and need cost[i]
gas to travel to station i+1. Return the starting station index if you can travel
around once, otherwise -1.

Key facts:
- If total gas < total cost, it's impossible from any start.
- If total gas >= total cost, a solution exists and can be found greedily.

Approach (greedy single pass):
Track `tank` for the current candidate start.
If tank ever goes negative at i, then no start in [start..i] can work (they would have
even less gas when reaching i), so set start = i+1 and reset tank.

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1

        start = 0
        tank = 0
        for i in range(len(gas)):
            tank += gas[i] - cost[i]
            if tank < 0:
                start = i + 1
                tank = 0
        return start


def run_tests() -> None:
    sol = Solution()
    assert sol.canCompleteCircuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]) == 3
    assert sol.canCompleteCircuit([2, 3, 4], [3, 4, 3]) == -1
    assert sol.canCompleteCircuit([5], [4]) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
