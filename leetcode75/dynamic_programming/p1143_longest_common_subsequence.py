"""1143. Longest Common Subsequence

Link: https://leetcode.com/problems/longest-common-subsequence/

DP (2D):
dp[i][j] = LCS length of text1[:i] and text2[:j]
Transition:
  if text1[i-1]==text2[j-1]: dp[i][j]=dp[i-1][j-1]+1
  else: dp[i][j]=max(dp[i-1][j], dp[i][j-1])

Space optimization: keep only previous row.
"""

from __future__ import annotations


class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        if not text1 or not text2:
            return 0
        # Ensure text2 is the shorter to reduce memory (optional).
        if len(text2) > len(text1):
            text1, text2 = text2, text1

        prev = [0] * (len(text2) + 1)
        for i in range(1, len(text1) + 1):
            cur = [0] * (len(text2) + 1)
            c1 = text1[i - 1]
            for j in range(1, len(text2) + 1):
                if c1 == text2[j - 1]:
                    cur[j] = prev[j - 1] + 1
                else:
                    cur[j] = max(prev[j], cur[j - 1])
            prev = cur
        return prev[-1]


def run_tests() -> None:
    sol = Solution()
    assert sol.longestCommonSubsequence("abcde", "ace") == 3
    assert sol.longestCommonSubsequence("abc", "abc") == 3
    assert sol.longestCommonSubsequence("abc", "def") == 0
    assert sol.longestCommonSubsequence("", "a") == 0


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


