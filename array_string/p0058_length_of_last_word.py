"""58. Length of Last Word

Link: https://leetcode.com/problems/length-of-last-word/

Problem:
Given a string consisting of words and spaces, return the length of the last word.

Approach (scan from end):
- Skip trailing spaces.
- Count characters until the next space or start of string.

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys


class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        i = len(s) - 1
        while i >= 0 and s[i] == " ":
            i -= 1
        length = 0
        while i >= 0 and s[i] != " ":
            length += 1
            i -= 1
        return length


def run_tests() -> None:
    sol = Solution()
    assert sol.lengthOfLastWord("Hello World") == 5
    assert sol.lengthOfLastWord("   fly me   to   the moon  ") == 4
    assert sol.lengthOfLastWord("luffy is still joyboy") == 6
    assert sol.lengthOfLastWord("a") == 1
    assert sol.lengthOfLastWord("a ") == 1
    assert sol.lengthOfLastWord("   ") == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
