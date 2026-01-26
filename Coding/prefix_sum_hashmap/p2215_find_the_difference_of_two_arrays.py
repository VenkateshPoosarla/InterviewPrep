"""2215. Find the Difference of Two Arrays

Link: https://leetcode.com/problems/find-the-difference-of-two-arrays/

Use sets:
- unique to nums1 not in nums2
- unique to nums2 not in nums1
Return as list of lists (order doesn't matter per problem statement).
"""

from __future__ import annotations

from typing import List


class Solution:
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        s1 = set(nums1)
        s2 = set(nums2)
        return [list(s1 - s2), list(s2 - s1)]


def run_tests() -> None:
    sol = Solution()

    out = sol.findDifference([1, 2, 3], [2, 4, 6])
    assert set(out[0]) == {1, 3}
    assert set(out[1]) == {4, 6}

    out = sol.findDifference([1, 2, 3, 3], [1, 1, 2, 2])
    assert set(out[0]) == {3}
    assert set(out[1]) == set()


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


