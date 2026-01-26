"""1456. Maximum Number of Vowels in a Substring of Given Length

Link: https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/

Fixed-size sliding window of length k.
Track number of vowels in current window; update by removing left char and adding right char.
"""

from __future__ import annotations


class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        vowels = set("aeiou")

        cur = 0
        for ch in s[:k]:
            cur += 1 if ch in vowels else 0
        best = cur

        for i in range(k, len(s)):
            cur += 1 if s[i] in vowels else 0
            cur -= 1 if s[i - k] in vowels else 0
            if cur > best:
                best = cur
                if best == k:  # can't do better than k
                    return best

        return best


def run_tests() -> None:
    sol = Solution()

    assert sol.maxVowels("abciiidef", 3) == 3
    assert sol.maxVowels("aeiou", 2) == 2
    assert sol.maxVowels("leetcode", 3) == 2
    assert sol.maxVowels("rhythms", 4) == 0
    assert sol.maxVowels("tryhard", 4) == 1


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


