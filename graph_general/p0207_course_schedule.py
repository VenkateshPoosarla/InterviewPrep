"""207. Course Schedule

Link: https://leetcode.com/problems/course-schedule/

Problem:
There are `numCourses` courses labeled [0..numCourses-1]. prerequisites[i] = [a, b]
means you must take course b before course a.
Return True if you can finish all courses (i.e., prerequisite graph has no cycles).

Approach (Kahn's algorithm / topological sort):
Build graph b -> a and indegree[a]++.
Push all nodes with indegree 0 into queue. Pop and relax edges, decrementing indegree.
If we can process all courses, there's no cycle.

Complexity:
- Time: O(V + E)
- Space: O(V + E)
"""

from __future__ import annotations

import sys
from collections import deque
from typing import Deque, List


class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph: List[List[int]] = [[] for _ in range(numCourses)]
        indeg = [0] * numCourses
        for a, b in prerequisites:
            graph[b].append(a)
            indeg[a] += 1

        q: Deque[int] = deque([i for i in range(numCourses) if indeg[i] == 0])
        taken = 0
        while q:
            v = q.popleft()
            taken += 1
            for nei in graph[v]:
                indeg[nei] -= 1
                if indeg[nei] == 0:
                    q.append(nei)
        return taken == numCourses


def run_tests() -> None:
    sol = Solution()
    assert sol.canFinish(2, [[1, 0]]) is True
    assert sol.canFinish(2, [[1, 0], [0, 1]]) is False
    assert sol.canFinish(1, []) is True


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
