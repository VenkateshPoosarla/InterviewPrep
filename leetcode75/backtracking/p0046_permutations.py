"""46. Permutations

Link: https://leetcode.com/problems/permutations/

Backtracking by swapping in-place:
Fix position i, try every choice from i..end by swapping, recurse, then swap back.
"""

from __future__ import annotations

from typing import List


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        out: List[List[int]] = []
        a = nums[:]  # don't mutate caller

        def backtrack(i: int) -> None:
            if i == len(a):
                out.append(a[:])
                return
            for j in range(i, len(a)):
                a[i], a[j] = a[j], a[i]
                backtrack(i + 1)
                a[i], a[j] = a[j], a[i]

        backtrack(0)
        return out


def run_tests() -> None:
    sol = Solution()

    out = sol.permute([1, 2, 3])
    assert len(out) == 6
    assert sorted(map(tuple, out)) == sorted(
        map(tuple, [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]])
    )
    assert sol.permute([0]) == [[0]]


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


