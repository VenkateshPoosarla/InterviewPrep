"""205. Isomorphic Strings

Link: https://leetcode.com/problems/isomorphic-strings/

Problem:
Two strings s and t are isomorphic if characters in s can be replaced to get t, with:
- Each character maps to exactly one other character
- No two characters map to the same character

Approach (two-way mapping):
Maintain mapping s->t and t->s to ensure bijection.
When a mapping conflicts, return False.

Complexity:
- Time: O(n)
- Space: O(k) for distinct characters
"""

from __future__ import annotations

import sys


class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        st: dict[str, str] = {}
        ts: dict[str, str] = {}
        for a, b in zip(s, t):
            if a in st and st[a] != b:
                return False
            if b in ts and ts[b] != a:
                return False
            st[a] = b
            ts[b] = a
        return True


def run_tests() -> None:
    sol = Solution()
    assert sol.isIsomorphic("egg", "add") is True
    assert sol.isIsomorphic("foo", "bar") is False
    assert sol.isIsomorphic("paper", "title") is True


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
