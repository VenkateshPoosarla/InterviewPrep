"""39. Combination Sum

Link: https://leetcode.com/problems/combination-sum/

Problem:
Given an array of distinct integers `candidates` and a target integer `target`, return
all unique combinations where chosen numbers sum to target.
You may reuse the same number unlimited times.

Approach (backtracking with index):
Sort candidates for pruning.
Backtrack with (start_index, remaining):
- For each candidate at i >= start_index:
  - choose it, recurse with same i (because reuse allowed), remaining - candidate
  - stop early if candidate > remaining

Complexity:
- Time: exponential in number of solutions
- Space: O(target/min_candidate) recursion depth + output
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        res: List[List[int]] = []
        path: list[int] = []

        def dfs(start: int, remaining: int) -> None:
            if remaining == 0:
                res.append(path.copy())
                return
            for i in range(start, len(candidates)):
                c = candidates[i]
                if c > remaining:
                    break
                path.append(c)
                dfs(i, remaining - c)
                path.pop()

        dfs(0, target)
        return res


def run_tests() -> None:
    sol = Solution()
    out = sol.combinationSum([2, 3, 6, 7], 7)
    assert sorted([sorted(x) for x in out]) == sorted([[2, 2, 3], [7]])

    out = sol.combinationSum([2, 3, 5], 8)
    assert sorted([sorted(x) for x in out]) == sorted([[2, 2, 2, 2], [2, 3, 3], [3, 5]])

    assert sol.combinationSum([2], 1) == []


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
