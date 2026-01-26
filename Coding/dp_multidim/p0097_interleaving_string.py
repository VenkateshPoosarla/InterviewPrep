"""97. Interleaving String

Link: https://leetcode.com/problems/interleaving-string/

Problem:
Return True if s3 is formed by an interleaving of s1 and s2, preserving the order of
characters in each string.

Approach (DP):
Let dp[i][j] mean s3[:i+j] can be formed from s1[:i] and s2[:j].
Transition:
dp[i][j] = (dp[i-1][j] and s1[i-1]==s3[i+j-1]) or (dp[i][j-1] and s2[j-1]==s3[i+j-1])
Use 1D rolling dp over j.

Complexity:
- Time: O(len(s1)*len(s2))
- Space: O(len(s2))
"""

from __future__ import annotations

import sys


class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        m, n = len(s1), len(s2)
        if m + n != len(s3):
            return False

        # dp[j] corresponds to dp[i][j] for current i
        dp = [False] * (n + 1)
        dp[0] = True
        for j in range(1, n + 1):
            dp[j] = dp[j - 1] and s2[j - 1] == s3[j - 1]

        for i in range(1, m + 1):
            dp[0] = dp[0] and s1[i - 1] == s3[i - 1]
            for j in range(1, n + 1):
                take_s1 = dp[j] and s1[i - 1] == s3[i + j - 1]
                take_s2 = dp[j - 1] and s2[j - 1] == s3[i + j - 1]
                dp[j] = take_s1 or take_s2
        return dp[n]


def run_tests() -> None:
    sol = Solution()
    assert sol.isInterleave("aabcc", "dbbca", "aadbbcbcac") is True
    assert sol.isInterleave("aabcc", "dbbca", "aadbbbaccc") is False
    assert sol.isInterleave("", "", "") is True
    assert sol.isInterleave("", "abc", "abc") is True
    assert sol.isInterleave("abc", "", "abc") is True


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
