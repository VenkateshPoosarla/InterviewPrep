"""1657. Determine if Two Strings Are Close

Link: https://leetcode.com/problems/determine-if-two-strings-are-close/

Allowed operations (from problem):
- Swap any two existing characters (reorder freely)
- Transform every occurrence of one existing character into another existing character, and vice versa

Implication:
- Both strings must have the same set of distinct characters.
- The multiset of character frequencies must match (because we can permute labels).

So we check:
1) set(word1) == set(word2)
2) sorted(Counter(word1).values()) == sorted(Counter(word2).values())
"""

from __future__ import annotations

from collections import Counter


class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        if len(word1) != len(word2):
            return False

        c1 = Counter(word1)
        c2 = Counter(word2)

        if set(c1.keys()) != set(c2.keys()):
            return False

        return sorted(c1.values()) == sorted(c2.values())


def run_tests() -> None:
    sol = Solution()

    assert sol.closeStrings("abc", "bca") is True
    assert sol.closeStrings("a", "aa") is False
    assert sol.closeStrings("cabbba", "abbccc") is True
    assert sol.closeStrings("cabbba", "aabbss") is False
    assert sol.closeStrings("", "") is True


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


