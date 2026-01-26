"""1071. Greatest Common Divisor of Strings

Link: https://leetcode.com/problems/greatest-common-divisor-of-strings/

Key fact:
If a gcd-string X exists, then str1 + str2 == str2 + str1.
Then the answer length is gcd(len(str1), len(str2)).

Visual:
    str1 = ABCABC
    str2 = ABC
    gcd lengths = gcd(6,3)=3 => "ABC"
"""

from __future__ import annotations

import math


class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        if str1 + str2 != str2 + str1:
            return ""
        g = math.gcd(len(str1), len(str2))
        return str1[:g]


def run_tests() -> None:
    s = Solution()

    assert s.gcdOfStrings("ABCABC", "ABC") == "ABC"
    assert s.gcdOfStrings("ABABAB", "ABAB") == "AB"
    assert s.gcdOfStrings("LEET", "CODE") == ""
    assert s.gcdOfStrings("A", "A") == "A"
    assert s.gcdOfStrings("AAAAAA", "AAA") == "AAA"


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


