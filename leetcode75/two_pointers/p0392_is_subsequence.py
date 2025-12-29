"""392. Is Subsequence

Link: https://leetcode.com/problems/is-subsequence/

Two pointers:
- i scans s
- j scans t
Advance j always; advance i when s[i] matches t[j].
s is subsequence iff i reaches len(s).

Visual:
  s = a c e
  t = a b c d e
      ^   ^   ^  => all matched in order
"""

from __future__ import annotations


class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        i = 0
        for ch in t:
            if i < len(s) and s[i] == ch:
                i += 1
                if i == len(s):
                    return True
        return i == len(s)


def run_tests() -> None:
    sol = Solution()

    assert sol.isSubsequence("abc", "ahbgdc") is True
    assert sol.isSubsequence("axc", "ahbgdc") is False
    assert sol.isSubsequence("", "ahbgdc") is True
    assert sol.isSubsequence("abc", "") is False
    assert sol.isSubsequence("aaaa", "aaaaa") is True


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


