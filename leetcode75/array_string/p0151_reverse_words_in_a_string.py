"""151. Reverse Words in a String

Link: https://leetcode.com/problems/reverse-words-in-a-string/

Pythonic approach:
- split() naturally collapses multiple spaces and trims ends
- reverse list
- join with single spaces
"""

from __future__ import annotations


class Solution:
    def reverseWords(self, s: str) -> str:
        parts = s.split()
        parts.reverse()
        return " ".join(parts)


def run_tests() -> None:
    sol = Solution()

    assert sol.reverseWords("the sky is blue") == "blue is sky the"
    assert sol.reverseWords("  hello world  ") == "world hello"
    assert sol.reverseWords("a good   example") == "example good a"
    assert sol.reverseWords("single") == "single"
    assert sol.reverseWords("   ") == ""


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


