"""345. Reverse Vowels of a String

Link: https://leetcode.com/problems/reverse-vowels-of-a-string/

Two pointers:
- i from left until vowel
- j from right until vowel
swap and continue

Vowels set includes both lowercase and uppercase.
"""

from __future__ import annotations


class Solution:
    def reverseVowels(self, s: str) -> str:
        vowels = set("aeiouAEIOU")
        a = list(s)
        i, j = 0, len(a) - 1
        while i < j:
            while i < j and a[i] not in vowels:
                i += 1
            while i < j and a[j] not in vowels:
                j -= 1
            if i < j:
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1
        return "".join(a)


def run_tests() -> None:
    s = Solution()

    assert s.reverseVowels("hello") == "holle"
    assert s.reverseVowels("leetcode") == "leotcede"
    assert s.reverseVowels("aA") == "Aa"
    assert s.reverseVowels("bbb") == "bbb"
    assert s.reverseVowels("") == ""


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


