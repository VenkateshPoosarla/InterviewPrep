"""46. Permutations

Link: https://leetcode.com/problems/permutations/

# Approach:
# 1. Fix position i, try every choice from i..end by swapping, recurse, then swap back.
# 2. Base case: if i == len(a), add the current permutation to the output.
# 3. Recursive case: for each choice from i..end, swap the current element with the choice, recurse, then swap back.
# 4. Return the output.
"""

from __future__ import annotations

from typing import List


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        self.out: List[List[int]] = []
        self.backtrack(nums, 0)
        return self.out
    # i is the index of the first element in the subarray to permute
    def backtrack(self, a: List[int], i: int) -> None:
            if i == len(a):
                self.out.append(a[:])
            for j in range(i, len(a)):
                a[i], a[j] = a[j], a[i] # swap
                self.backtrack(a, i + 1)
                a[i], a[j] = a[j], a[i] # swap back 


def run_tests() -> None:
    sol = Solution()

    out = sol.permute([1, 2, 3])
    print(out)
    assert len(out) == 6
    assert sorted(map(tuple, out)) == sorted(
        map(tuple, [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]])
    )
    assert sol.permute([0]) == [[0]]


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


