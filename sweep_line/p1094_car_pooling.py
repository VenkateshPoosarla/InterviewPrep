"""1094. Car Pooling

Link: https://leetcode.com/problems/car-pooling/

Pattern: Sweep line with difference array over locations.

For each trip [num, start, end):
  diff[start] += num
  diff[end]   -= num
Prefix sum gives current passengers at each location.
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        # constraints: locations are in [0..1000]
        MAX_POS = 1000
        diff = [0] * (MAX_POS + 2)
        for num, start, end in trips:
            diff[int(start)] += int(num)
            diff[int(end)] -= int(num)

        cur = 0
        for x in diff:
            cur += x
            if cur > capacity:
                return False
        return True


def run_tests() -> None:
    sol = Solution()
    assert sol.carPooling([[2, 1, 5], [3, 3, 7]], 4) is False
    assert sol.carPooling([[2, 1, 5], [3, 3, 7]], 5) is True
    assert sol.carPooling([[3, 2, 7], [3, 7, 9], [8, 3, 9]], 11) is True


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


