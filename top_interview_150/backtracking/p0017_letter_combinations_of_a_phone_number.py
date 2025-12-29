"""17. Letter Combinations of a Phone Number

Link: https://leetcode.com/problems/letter-combinations-of-a-phone-number/

Problem:
Given a string of digits 2-9 inclusive, return all possible letter combinations that
the number could represent (phone keypad mapping).

Approach (backtracking):
At each digit, try each possible letter and recurse to the next digit.

Complexity:
- Time: O(3^n) to O(4^n) depending on digits
- Space: O(n) recursion depth + output
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        phone = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }

        out: List[str] = []
        path: list[str] = []

        def dfs(i: int) -> None:
            if i == len(digits):
                out.append("".join(path))
                return
            for ch in phone[digits[i]]:
                path.append(ch)
                dfs(i + 1)
                path.pop()

        dfs(0)
        return out


def run_tests() -> None:
    sol = Solution()
    assert sorted(sol.letterCombinations("23")) == sorted(
        ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]
    )
    assert sol.letterCombinations("") == []
    assert sorted(sol.letterCombinations("2")) == ["a", "b", "c"]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
