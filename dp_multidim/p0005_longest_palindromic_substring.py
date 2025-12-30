"""5. Longest Palindromic Substring

Link: https://leetcode.com/problems/longest-palindromic-substring/

Problem:
Return the longest palindromic substring in `s`.

Approach (expand around center):
Every palindrome has a center:
- odd length: center at i
- even length: center between i and i+1
Expand while characters match and track the best window.

Complexity:
- Time: O(n^2)
- Space: O(1)
"""

from __future__ import annotations

import sys


class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return ""

        best_l, best_r = 0, 0

        def expand(l: int, r: int) -> None:
            nonlocal best_l, best_r
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            # palindrome is s[l+1:r]
            if r - (l + 1) > best_r - best_l + 1:
                best_l, best_r = l + 1, r - 1

        for i in range(len(s)):
            expand(i, i)       # odd
            expand(i, i + 1)   # even

        return s[best_l : best_r + 1]


def run_tests() -> None:
    sol = Solution()
    assert sol.longestPalindrome("babad") in {"bab", "aba"}
    assert sol.longestPalindrome("cbbd") == "bb"
    assert sol.longestPalindrome("a") == "a"
    assert sol.longestPalindrome("ac") in {"a", "c"}


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
