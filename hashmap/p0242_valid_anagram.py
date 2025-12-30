"""242. Valid Anagram

Link: https://leetcode.com/problems/valid-anagram/

Problem:
Return True if t is an anagram of s (same letters with same multiplicity).

Approach:
Compare character counts.

Complexity:
- Time: O(n)
- Space: O(1) (bounded by alphabet size)
"""

from __future__ import annotations

import sys
from collections import Counter


class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return Counter(s) == Counter(t)


def run_tests() -> None:
    sol = Solution()
    assert sol.isAnagram("anagram", "nagaram") is True
    assert sol.isAnagram("rat", "car") is False
    assert sol.isAnagram("", "") is True


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
