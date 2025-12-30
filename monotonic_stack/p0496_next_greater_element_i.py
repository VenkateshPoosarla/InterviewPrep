"""496. Next Greater Element I

Link: https://leetcode.com/problems/next-greater-element-i/

For each element in nums1, find the next greater element to its right in nums2.

Monotonic decreasing stack over nums2:
- Pop smaller elements; when we see a bigger x, it is the next greater for popped ones.
Build a map: next_greater[val] = next greater value (or -1 if none).
Then answer for nums1 is lookup in map.
"""

from __future__ import annotations

from typing import Dict, List


class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nxt: Dict[int, int] = {}
        st: list[int] = []

        for x in nums2:
            while st and x > st[-1]:
                nxt[st.pop()] = x
            st.append(x)

        for x in st:
            nxt[x] = -1

        return [nxt[x] for x in nums1]


def run_tests() -> None:
    sol = Solution()
    assert sol.nextGreaterElement([4, 1, 2], [1, 3, 4, 2]) == [-1, 3, -1]
    assert sol.nextGreaterElement([2, 4], [1, 2, 3, 4]) == [3, -1]
    assert sol.nextGreaterElement([], [1, 2]) == []


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


