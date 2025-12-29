"""77. Combinations

Link: https://leetcode.com/problems/combinations/

Backtracking: build combinations of size k from numbers 1..n.
Use pruning: if remaining numbers are insufficient, stop early.
"""

from __future__ import annotations

from typing import List


class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        out: List[List[int]] = []
        path: list[int] = []

        def backtrack(start: int) -> None:
            if len(path) == k:
                out.append(path[:])
                return

            # Choose next number i. Need (k-len(path)) numbers total.
            need = k - len(path)
            # Last possible start that still leaves enough numbers:
            # i can go up to n - need + 1
            for i in range(start, n - need + 2):
                path.append(i)
                backtrack(i + 1)
                path.pop()

        if k == 0:
            return [[]]
        if k > n:
            return []
        backtrack(1)
        return out


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


