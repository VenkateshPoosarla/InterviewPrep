"""502. IPO

Link: https://leetcode.com/problems/ipo/

Problem:
You have `k` projects. Each project i requires capital[i] and yields profit[i].
Starting with capital `w`, you can do at most k projects. After doing a project, your
capital increases by its profit. Return the maximum capital after at most k projects.

Approach (sort by required capital + max-heap of profits):
Sort projects by capital requirement.
For each of k rounds:
- Push profits of all projects whose capital <= current_w into a max-heap.
- If heap empty, cannot do any project -> break.
- Pop the max profit and add to w.

Complexity:
- Time: O(n log n + k log n)
- Space: O(n)
"""

from __future__ import annotations

import heapq
import sys
from typing import List, Tuple


class Solution:
    def findMaximizedCapital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        projects: List[Tuple[int, int]] = sorted(zip(capital, profits))
        i = 0
        max_heap: list[int] = []  # store negative profits

        for _ in range(k):
            while i < len(projects) and projects[i][0] <= w:
                heapq.heappush(max_heap, -projects[i][1])
                i += 1
            if not max_heap:
                break
            w += -heapq.heappop(max_heap)
        return w


def run_tests() -> None:
    sol = Solution()
    assert sol.findMaximizedCapital(2, 0, [1, 2, 3], [0, 1, 1]) == 4
    assert sol.findMaximizedCapital(1, 0, [1, 2, 3], [1, 1, 2]) == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
