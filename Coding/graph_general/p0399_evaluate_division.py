"""399. Evaluate Division

Link: https://leetcode.com/problems/evaluate-division/

Problem:
Given equations like a / b = value, answer queries x / y.
If x or y is unknown or no path exists, answer -1.0.

Approach (graph + DFS/BFS per query):
Build a directed weighted graph:
- a -> b with weight value
- b -> a with weight 1/value

For each query, do BFS/DFS to find a path from x to y, multiplying weights along the path.

Complexity:
- Build: O(E)
- Each query: O(V + E) in worst case
"""

from __future__ import annotations

import sys
from collections import deque
from typing import Deque, Dict, List, Tuple


class Solution:
    def calcEquation(
        self,
        equations: List[List[str]],
        values: List[float],
        queries: List[List[str]],
    ) -> List[float]:
        graph: Dict[str, List[Tuple[str, float]]] = {}
        for (a, b), v in zip(equations, values):
            graph.setdefault(a, []).append((b, v))
            graph.setdefault(b, []).append((a, 1.0 / v))

        def solve_query(src: str, dst: str) -> float:
            if src not in graph or dst not in graph:
                return -1.0
            if src == dst:
                return 1.0
            q: Deque[Tuple[str, float]] = deque([(src, 1.0)])
            seen = {src}
            while q:
                node, acc = q.popleft()
                for nei, w in graph.get(node, []):
                    if nei in seen:
                        continue
                    nxt = acc * w
                    if nei == dst:
                        return nxt
                    seen.add(nei)
                    q.append((nei, nxt))
            return -1.0

        return [solve_query(x, y) for x, y in queries]


def run_tests() -> None:
    sol = Solution()
    equations = [["a", "b"], ["b", "c"]]
    values = [2.0, 3.0]
    queries = [["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"]]
    out = sol.calcEquation(equations, values, queries)
    assert out[0] == 6.0
    assert out[1] == 0.5
    assert out[2] == -1.0
    assert out[3] == 1.0
    assert out[4] == -1.0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
