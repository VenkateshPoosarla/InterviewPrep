"""290. Word Pattern

Link: https://leetcode.com/problems/word-pattern/

Problem:
Given a pattern string and a space-separated string `s`, return True if `s` follows the
same pattern:
- Each pattern character maps to exactly one word
- No two pattern characters map to the same word

Approach (two-way mapping):
Split s into words; lengths must match.
Maintain char->word and word->char maps and ensure consistency.

Complexity:
- Time: O(n)
- Space: O(k)
"""

from __future__ import annotations

import sys


class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        words = s.split()
        if len(words) != len(pattern):
            return False

        cw: dict[str, str] = {}
        wc: dict[str, str] = {}
        for ch, w in zip(pattern, words):
            if ch in cw and cw[ch] != w:
                return False
            if w in wc and wc[w] != ch:
                return False
            cw[ch] = w
            wc[w] = ch
        return True


def run_tests() -> None:
    sol = Solution()
    assert sol.wordPattern("abba", "dog cat cat dog") is True
    assert sol.wordPattern("abba", "dog cat cat fish") is False
    assert sol.wordPattern("aaaa", "dog cat cat dog") is False
    assert sol.wordPattern("abba", "dog dog dog dog") is False


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
