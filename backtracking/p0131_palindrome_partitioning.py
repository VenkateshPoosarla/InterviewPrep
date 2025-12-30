"""131. Palindrome Partitioning

Link: https://leetcode.com/problems/palindrome-partitioning/

Backtracking over cut positions.
At each start index, try every end index that forms a palindrome, append it, and recurse.

Optimization: precompute palindromes dp[i][j] (optional).
Here we use a simple O(n) palindrome check (good enough for interviewprep),
but still keep code clean.
"""

from __future__ import annotations

from typing import List


class Solution:
    def partition(self, s: str) -> List[List[str]]:
        out: List[List[str]] = []
        path: list[str] = []

        def is_pal(l: int, r: int) -> bool:
            while l < r:
                if s[l] != s[r]:
                    return False
                l += 1
                r -= 1
            return True

        def backtrack(start: int) -> None:
            if start == len(s):
                out.append(path[:])
                return
            for end in range(start, len(s)):
                if is_pal(start, end):
                    path.append(s[start : end + 1])
                    backtrack(end + 1)
                    path.pop()

        backtrack(0)
        return out


def run_tests() -> None:
    sol = Solution()

    out = sol.partition("aab")
    assert sorted(map(tuple, out)) == sorted(map(tuple, [["a", "a", "b"], ["aa", "b"]]))

    out = sol.partition("a")
    assert out == [["a"]]

    out = sol.partition("")
    assert out == [[]]


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


