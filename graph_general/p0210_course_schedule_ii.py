"""210. Course Schedule II

Link: https://leetcode.com/problems/course-schedule-ii/

Problem:
Return an ordering of courses you should take to finish all courses.
If it's impossible, return [].

Approach (Kahn's algorithm):
Same graph construction as 207, but record the order as we pop nodes with indegree 0.
If we don't process all nodes, there is a cycle -> return [].

Complexity:
- Time: O(V + E)
- Space: O(V + E)
"""

from __future__ import annotations

import sys
from collections import deque
from typing import Deque, List


class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph: List[List[int]] = [[] for _ in range(numCourses)]
        indeg = [0] * numCourses
        for a, b in prerequisites:
            graph[b].append(a)
            indeg[a] += 1

        q: Deque[int] = deque([i for i in range(numCourses) if indeg[i] == 0])
        order: List[int] = []
        while q:
            v = q.popleft()
            order.append(v)
            for nei in graph[v]:
                indeg[nei] -= 1
                if indeg[nei] == 0:
                    q.append(nei)

        return order if len(order) == numCourses else []


def run_tests() -> None:
    sol = Solution()
    out = sol.findOrder(2, [[1, 0]])
    assert out == [0, 1]

    out = sol.findOrder(4, [[1, 0], [2, 0], [3, 1], [3, 2]])
    # multiple valid outputs; verify prerequisites satisfied
    pos = {c: i for i, c in enumerate(out)}
    assert set(out) == {0, 1, 2, 3}
    assert pos[0] < pos[1]
    assert pos[0] < pos[2]
    assert pos[1] < pos[3]
    assert pos[2] < pos[3]

    assert sol.findOrder(2, [[1, 0], [0, 1]]) == []


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
