"""338. Counting Bits

Link: https://leetcode.com/problems/counting-bits/

Return ans[i] = number of 1-bits in i for i in [0..n].

DP trick:
ans[i] = ans[i >> 1] + (i & 1)
Because shifting right removes the last bit.
"""

from __future__ import annotations

from typing import List


class Solution:
    def countBits(self, n: int) -> List[int]:
        ans = [0] * (n + 1)
        for i in range(1, n + 1):
            ans[i] = ans[i >> 1] + (i & 1)
        return ans


def run_tests() -> None:
    sol = Solution()
    assert sol.countBits(2) == [0, 1, 1]
    assert sol.countBits(5) == [0, 1, 1, 2, 1, 2]
    assert sol.countBits(0) == [0]


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


