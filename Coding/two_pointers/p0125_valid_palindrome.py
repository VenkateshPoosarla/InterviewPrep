"""125. Valid Palindrome

Link: https://leetcode.com/problems/valid-palindrome/

Problem:
Return True if `s` is a palindrome after converting to lowercase and removing all
non-alphanumeric characters.

Approach (two pointers):
Use two indices l/r. Move l forward until it points to alnum, move r backward until alnum.
Compare lowercase characters; if any mismatch, not a palindrome.

Complexity:
- Time: O(n)
- Space: O(1)
"""

from __future__ import annotations

import sys


class Solution:
    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1
        while l < r:
            while l < r and not s[l].isalnum():
                l += 1
            while l < r and not s[r].isalnum():
                r -= 1
            if s[l].lower() != s[r].lower():
                return False
            l += 1
            r -= 1
        return True


def run_tests() -> None:
    sol = Solution()
    assert sol.isPalindrome("A man, a plan, a canal: Panama") is True
    assert sol.isPalindrome("race a car") is False
    assert sol.isPalindrome(" ") is True
    assert sol.isPalindrome("0P") is False


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
