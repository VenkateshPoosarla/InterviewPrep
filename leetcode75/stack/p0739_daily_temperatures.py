"""739. Daily Temperatures

Link: https://leetcode.com/problems/daily-temperatures/

Monotonic decreasing stack of indices (temperatures strictly decreasing).
When we see a warmer day, we resolve previous days.

Visual:
  temps: 73 74 75 71 69 72 76 73
  stack holds indices of unresolved days with decreasing temps.
"""

from __future__ import annotations

from typing import List


class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0] * n
        st: list[int] = []  # indices

        for i, t in enumerate(temperatures):
            while st and t > temperatures[st[-1]]:
                j = st.pop()
                ans[j] = i - j
            st.append(i)

        return ans


def run_tests() -> None:
    sol = Solution()

    assert sol.dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
    assert sol.dailyTemperatures([30, 40, 50, 60]) == [1, 1, 1, 0]
    assert sol.dailyTemperatures([30, 60, 90]) == [1, 1, 0]
    assert sol.dailyTemperatures([]) == []
    assert sol.dailyTemperatures([50]) == [0]


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


