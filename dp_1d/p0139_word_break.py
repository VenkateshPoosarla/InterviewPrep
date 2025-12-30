"""139. Word Break

Link: https://leetcode.com/problems/word-break/

Problem:
Given a string s and a dictionary of strings wordDict, return True if s can be
segmented into a space-separated sequence of one or more dictionary words.

Approach (DP over prefix):
dp[i] = True if s[:i] can be segmented.
Transition:
  dp[i] = any(dp[j] and s[j:i] in wordSet) for j < i
We can optimize by only trying word lengths present in dict.

Complexity:
- Time: O(n * L) where L is number of distinct word lengths (worst O(n^2))
- Space: O(n)
"""

from __future__ import annotations

import sys
from typing import List, Set


class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        word_set: Set[str] = set(wordDict)
        if not word_set:
            return False
        lens = sorted({len(w) for w in word_set})

        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True

        for i in range(1, n + 1):
            for L in lens:
                if L > i:
                    break
                if dp[i - L] and s[i - L : i] in word_set:
                    dp[i] = True
                    break
        return dp[n]


def run_tests() -> None:
    sol = Solution()
    assert sol.wordBreak("leetcode", ["leet", "code"]) is True
    assert sol.wordBreak("applepenapple", ["apple", "pen"]) is True
    assert sol.wordBreak("catsandog", ["cats", "dog", "sand", "and", "cat"]) is False


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
