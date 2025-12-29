"""72. Edit Distance

Link: https://leetcode.com/problems/edit-distance/

Compute Levenshtein distance between word1 and word2:
minimum operations (insert, delete, replace) to transform word1 -> word2.

DP:
dp[i][j] = min edits for word1[:i] -> word2[:j]
Base:
dp[i][0]=i (delete all)
dp[0][j]=j (insert all)
Transition:
if chars equal: dp[i][j] = dp[i-1][j-1]
else:
  dp[i][j] = 1 + min(
      dp[i-1][j],   # delete
      dp[i][j-1],   # insert
      dp[i-1][j-1], # replace
  )

Space: O(min(m,n)) using rolling rows.
"""

from __future__ import annotations


class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        # Ensure word2 is shorter for less memory.
        if len(word2) > len(word1):
            word1, word2 = word2, word1

        m, n = len(word1), len(word2)
        prev = list(range(n + 1))  # dp[0][j]

        for i in range(1, m + 1):
            cur = [i] + [0] * n
            c1 = word1[i - 1]
            for j in range(1, n + 1):
                if c1 == word2[j - 1]:
                    cur[j] = prev[j - 1]
                else:
                    cur[j] = 1 + min(prev[j], cur[j - 1], prev[j - 1])
            prev = cur

        return prev[n]


def run_tests() -> None:
    sol = Solution()
    assert sol.minDistance("horse", "ros") == 3
    assert sol.minDistance("intention", "execution") == 5
    assert sol.minDistance("", "") == 0
    assert sol.minDistance("a", "") == 1
    assert sol.minDistance("", "abc") == 3


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


