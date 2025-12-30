"""169. Majority Element

Link: https://leetcode.com/problems/majority-element/

Problem:
Find the element that appears more than ⌊n/2⌋ times in the array.
It is guaranteed to exist.

Approach (Boyer–Moore Voting):
Maintain a candidate and a counter.
- If counter is 0, set candidate = current.
- If current == candidate, counter++ else counter--.
Intuition: majority element "cancels out" all other elements combined.

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        candidate = 0
        count = 0
        for x in nums:
            if count == 0:
                candidate = x
                count = 1
            elif x == candidate:
                count += 1
            else:
                count -= 1
        return candidate


def run_tests() -> None:
    sol = Solution()
    assert sol.majorityElement([3, 2, 3]) == 3
    assert sol.majorityElement([2, 2, 1, 1, 1, 2, 2]) == 2
    assert sol.majorityElement([1]) == 1


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
