"""219. Contains Duplicate II

Link: https://leetcode.com/problems/contains-duplicate-ii/

Problem:
Return True if there are two distinct indices i and j such that:
- nums[i] == nums[j]
- |i - j| <= k

Approach (sliding set of size k):
Maintain a set of the last k elements we've seen.
For each x:
- if x already in set, we found duplicates within distance k
- add x
- if window is bigger than k, remove nums[i-k]

Complexity:
- Time: O(n)
- Space: O(k)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        window: set[int] = set()
        for i, x in enumerate(nums):
            if x in window:
                return True
            window.add(x)
            if i >= k:
                window.remove(nums[i - k])
        return False


def run_tests() -> None:
    sol = Solution()
    assert sol.containsNearbyDuplicate([1, 2, 3, 1], 3) is True
    assert sol.containsNearbyDuplicate([1, 0, 1, 1], 1) is True
    assert sol.containsNearbyDuplicate([1, 2, 3, 1, 2, 3], 2) is False


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
