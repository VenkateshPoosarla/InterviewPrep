"""28. Implement strStr() (Find the Index of the First Occurrence in a String)

Link: https://leetcode.com/problems/implement-strstr/

Problem:
Return the index of the first occurrence of `needle` in `haystack`, or -1 if not found.

Approach (KMP):
Knuth–Morris–Pratt avoids re-checking characters by precomputing an LPS array:
- lps[i] = length of the longest proper prefix of needle[:i+1] that is also a suffix.

Then scan haystack with pointers i (haystack), j (needle):
- If match: advance both
- If mismatch and j>0: j = lps[j-1] (shift pattern using LPS)
- Else mismatch at j==0: advance i

Complexity:
- Time: O(n + m)
- Space: O(m)
"""

from __future__ import annotations

import sys


class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if needle == "":
            return 0

        # Build LPS for needle.
        lps = [0] * len(needle)
        length = 0  # length of current longest prefix-suffix
        i = 1
        while i < len(needle):
            if needle[i] == needle[length]:
                length += 1
                lps[i] = length
                i += 1
            elif length > 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

        # Scan haystack.
        i = 0
        j = 0
        while i < len(haystack):
            if haystack[i] == needle[j]:
                i += 1
                j += 1
                if j == len(needle):
                    return i - j
            elif j > 0:
                j = lps[j - 1]
            else:
                i += 1
        return -1


def run_tests() -> None:
    sol = Solution()
    assert sol.strStr("sadbutsad", "sad") == 0
    assert sol.strStr("sadbutsad", "but") == 3
    assert sol.strStr("leetcode", "leeto") == -1
    assert sol.strStr("aaaaa", "bba") == -1
    assert sol.strStr("a", "a") == 0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
