"""1768. Merge Strings Alternately

Link: https://leetcode.com/problems/merge-strings-alternately/

Idea:
Walk both strings with indices i/j, alternately append chars while either remains.

Visual:
    word1 = a b c
    word2 = p q r s
    take:  a p b q c r   + leftover: s  => apbqcrs
"""

from __future__ import annotations


class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        i = 0
        j = 0
        out: list[str] = []
        take_from_first = True

        while i < len(word1) or j < len(word2):
            if take_from_first:
                if i < len(word1):
                    out.append(word1[i])
                    i += 1
                elif j < len(word2):
                    out.append(word2[j])
                    j += 1
            else:
                if j < len(word2):
                    out.append(word2[j])
                    j += 1
                elif i < len(word1):
                    out.append(word1[i])
                    i += 1

            take_from_first = not take_from_first

        return "".join(out)


def run_tests() -> None:
    s = Solution()

    assert s.mergeAlternately("abc", "pqr") == "apbqcr"
    assert s.mergeAlternately("ab", "pqrs") == "apbqrs"
    assert s.mergeAlternately("abcd", "pq") == "apbqcd"
    assert s.mergeAlternately("", "") == ""
    assert s.mergeAlternately("", "xyz") == "xyz"
    assert s.mergeAlternately("xyz", "") == "xyz"


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


