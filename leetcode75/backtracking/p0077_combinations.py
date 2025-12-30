"""77. Combinations

Link: https://leetcode.com/problems/combinations/

Backtracking: build combinations of size k from numbers 1..n.
Use pruning: if remaining numbers are insufficient, stop early.
"""

from __future__ import annotations

from typing import List


class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        self.out: List[List[int]] = []
        self.n = n
        self.k = k
        path: list[int] = []
        self.backtrack(1, path)
        return self.out    
        
    def backtrack(self, start: int, path: list[int]) -> None:
        if len(path) == self.k:
            self.out.append(path[:])
            return
        for i in range(start, self.n + 1):
            path.append(i)
            self.backtrack(i + 1, path)
            path.pop()


def run_tests() -> None:
    sol = Solution()

    out = sol.combine(4, 2)
    assert sorted(map(tuple, out)) == sorted(map(tuple, [[2, 4], [3, 4], [2, 3], [1, 2], [1, 3], [1, 4]]))
    assert sol.combine(1, 1) == [[1]]
    assert sol.combine(5, 0) == [[]]
    assert sol.combine(2, 3) == []


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


