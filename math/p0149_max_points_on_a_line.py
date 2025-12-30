"""149. Max Points on a Line

Link: https://leetcode.com/problems/max-points-on-a-line/

Problem:
Given points on a 2D plane, return the maximum number of points that lie on the same straight line.

Approach (hash slopes per anchor):
For each anchor point i, compute slopes to all j>i and count identical slopes.
To avoid floating precision:
- represent slope as a reduced fraction (dy/g, dx/g) with a canonical sign
Handle duplicates (same point) separately.

Complexity:
- Time: O(n^2)
- Space: O(n)
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Tuple


class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        n = len(points)
        if n <= 2:
            return n

        best = 1
        for i in range(n):
            slopes: Dict[Tuple[int, int], int] = {}
            dup = 0
            local_best = 0
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    dup += 1
                    continue
                g = math.gcd(dx, dy)
                dx //= g
                dy //= g
                # canonicalize sign: keep dx positive; if dx==0, keep dy = 1/-1
                if dx < 0:
                    dx = -dx
                    dy = -dy
                if dx == 0:
                    dy = 1 if dy > 0 else -1
                if dy == 0:
                    dx = 1
                key = (dy, dx)
                slopes[key] = slopes.get(key, 0) + 1
                local_best = max(local_best, slopes[key])
            best = max(best, local_best + dup + 1)
        return best


def run_tests() -> None:
    sol = Solution()
    assert sol.maxPoints([[1, 1], [2, 2], [3, 3]]) == 3
    assert sol.maxPoints([[1, 1], [3, 2], [5, 3], [4, 1], [2, 3], [1, 4]]) == 4
    assert sol.maxPoints([[0, 0]]) == 1
    assert sol.maxPoints([[0, 0], [0, 0], [1, 1]]) == 3


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
