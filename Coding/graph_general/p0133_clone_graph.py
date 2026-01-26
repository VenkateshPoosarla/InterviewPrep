"""133. Clone Graph

Link: https://leetcode.com/problems/clone-graph/

Problem:
Given a reference to a node in an undirected connected graph, return a deep copy (clone)
of the graph.

Each node has:
- val: int
- neighbors: list[Node]

Approach (DFS + hash map):
Use a dict old_node -> new_node to avoid re-cloning and to break cycles.
DFS:
- if node already cloned, return it
- else create clone, store in dict, then clone neighbors recursively

Complexity:
- Time: O(V + E)
- Space: O(V)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Node:
    val: int
    neighbors: List["Node"] = field(default_factory=list)


class Solution:
    def cloneGraph(self, node: Optional[Node]) -> Optional[Node]:
        if node is None:
            return None

        clones: Dict[int, Node] = {}
        # Key by object identity isn't hashable with dataclass default; use id(node) or store by node itself.
        # We'll key by id(node) and store neighbor ids during traversal.
        seen: Dict[int, Node] = {}

        def dfs(cur: Node) -> Node:
            key = id(cur)
            if key in seen:
                return seen[key]
            copy = Node(cur.val)
            seen[key] = copy
            copy.neighbors = [dfs(nb) for nb in cur.neighbors]
            return copy

        return dfs(node)


def _serialize(node: Optional[Node]) -> List[List[int]]:
    """Serialize to adjacency list (LeetCode-style) by BFS, using node values as labels."""
    if node is None:
        return []
    # LeetCode's standard graph here uses 1..n labels and connected graph.
    from collections import deque

    q = deque([node])
    seen = {id(node)}
    order: List[Node] = []
    while q:
        cur = q.popleft()
        order.append(cur)
        for nb in cur.neighbors:
            if id(nb) not in seen:
                seen.add(id(nb))
                q.append(nb)
    # map node object to index by BFS order (not necessarily value order)
    idx = {id(n): i for i, n in enumerate(order)}
    adj: List[List[int]] = [[] for _ in order]
    for i, n in enumerate(order):
        adj[i] = [idx[id(nb)] for nb in n.neighbors]
    return adj


def run_tests() -> None:
    sol = Solution()

    # Build a 4-node cycle-ish graph:
    # 1: [2,4]
    # 2: [1,3]
    # 3: [2,4]
    # 4: [1,3]
    n1 = Node(1)
    n2 = Node(2)
    n3 = Node(3)
    n4 = Node(4)
    n1.neighbors = [n2, n4]
    n2.neighbors = [n1, n3]
    n3.neighbors = [n2, n4]
    n4.neighbors = [n1, n3]

    clone = sol.cloneGraph(n1)
    assert clone is not None and clone is not n1
    assert _serialize(clone) == _serialize(n1)

    assert sol.cloneGraph(None) is None


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
