"""14. Longest Common Prefix

Link: https://leetcode.com/problems/longest-common-prefix/

Problem:
Given a list of strings, return the longest common prefix among them.

Approach (min/max trick):
If you sort the strings, the common prefix of the whole set equals the common prefix
of the first and last strings (lexicographically), because those two are the most
different.

Complexity:
- Time: O(n log n * L) due to sort comparisons (L = avg string length)
- Space: O(1) extra (ignoring sort internals)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""
        strs.sort()
        a, b = strs[0], strs[-1]
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        return a[:i]


def run_tests() -> None:
    sol = Solution()
    assert sol.longestCommonPrefix(["flower", "flow", "flight"]) == "fl"
    assert sol.longestCommonPrefix(["dog", "racecar", "car"]) == ""
    assert sol.longestCommonPrefix(["a"]) == "a"
    assert sol.longestCommonPrefix([]) == ""
    assert sol.longestCommonPrefix(["", ""]) == ""


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
