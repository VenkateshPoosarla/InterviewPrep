"""459. Repeated Substring Pattern

Link: https://leetcode.com/problems/repeated-substring-pattern/

Pattern: KMP prefix-function (LPS array).

If s has length n and l = lps[n-1], then s is repeating iff:
- l > 0 and n % (n - l) == 0
"""

from __future__ import annotations

import sys


class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        n = len(s)
        if n <= 1:
            return False

        lps = [0] * n
        j = 0
        for i in range(1, n):
            while j > 0 and s[i] != s[j]:
                j = lps[j - 1]
            if s[i] == s[j]:
                j += 1
                lps[i] = j

        l = lps[-1]
        return l > 0 and n % (n - l) == 0


def run_tests() -> None:
    sol = Solution()
    assert sol.repeatedSubstringPattern("abab") is True
    assert sol.repeatedSubstringPattern("aba") is False
    assert sol.repeatedSubstringPattern("abcabcabcabc") is True
    assert sol.repeatedSubstringPattern("a") is False


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


