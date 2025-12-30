"""9. Palindrome Number

Link: https://leetcode.com/problems/palindrome-number/

Problem:
Return True if an integer is a palindrome (reads the same backward as forward).

Approach (reverse half):
Negative numbers are not palindromes. Numbers ending with 0 are not palindromes unless the
number is 0.
Reverse digits until reversed >= remaining number. For even length: x == rev.
For odd length: x == rev//10 (drop the middle digit).

Complexity:
- Time: O(log10(n))
- Space: O(1)
"""

from __future__ import annotations

import sys


class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        if x != 0 and x % 10 == 0:
            return False

        rev = 0
        while x > rev:
            x, d = divmod(x, 10)
            rev = rev * 10 + d
        return x == rev or x == rev // 10


def run_tests() -> None:
    sol = Solution()
    assert sol.isPalindrome(121) is True
    assert sol.isPalindrome(-121) is False
    assert sol.isPalindrome(10) is False
    assert sol.isPalindrome(0) is True


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
