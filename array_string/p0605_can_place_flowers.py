"""605. Can Place Flowers

Link: https://leetcode.com/problems/can-place-flowers/

We can greedily plant from left-to-right.

Rule: we can plant at i if flowerbed[i]==0 and neighbors (i-1,i+1) are 0 or out of bounds.

Visual:
  bed:  1 0 0 0 1
        ^ plant here => 1 0 1 0 1
"""

from __future__ import annotations

from typing import List


class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        if n <= 0:
            return True

        bed = flowerbed[:]  # do not mutate caller in tests
        for i in range(len(bed)):
            if bed[i] == 1:
                continue

            left_empty = (i == 0) or (bed[i - 1] == 0)
            right_empty = (i == len(bed) - 1) or (bed[i + 1] == 0)
            if left_empty and right_empty:
                bed[i] = 1
                n -= 1
                if n == 0:
                    return True

        return n == 0


def run_tests() -> None:
    s = Solution()

    assert s.canPlaceFlowers([1, 0, 0, 0, 1], 1) is True
    assert s.canPlaceFlowers([1, 0, 0, 0, 1], 2) is False
    assert s.canPlaceFlowers([0], 1) is True
    assert s.canPlaceFlowers([0, 0, 0, 0, 0], 3) is True
    assert s.canPlaceFlowers([0, 1, 0], 1) is False
    assert s.canPlaceFlowers([0, 0], 1) is True
    assert s.canPlaceFlowers([0, 0], 2) is False


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


