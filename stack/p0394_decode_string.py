"""394. Decode String

Link: https://leetcode.com/problems/decode-string/

Grammar:
  encoded_string := plain | encoded_string plain
  plain := letters | k[encoded_string]

Stack approach:
Maintain a stack of (previous_string, repeat_count) when encountering '['.
Build current string as we scan.

Visual:
  s = "3[a2[c]]"
    num=3, push ("",3), cur=""
    'a' => cur="a"
    num=2, push ("a",2), cur=""
    'c' => cur="c"
    ']' => pop ("a",2) => cur="a"+"c"*2="acc"
    ']' => pop ("",3) => cur=""+"acc"*3="accaccacc"
"""

from __future__ import annotations


class Solution:
    def decodeString(self, s: str) -> str:
        stack: list[tuple[str, int]] = []
        cur = ""
        num = 0

        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(ch)
            elif ch == "[":
                stack.append((cur, num))
                cur = ""
                num = 0
            elif ch == "]":
                prev, k = stack.pop()
                cur = prev + cur * k
            else:
                cur += ch

        return cur


def run_tests() -> None:
    sol = Solution()

    assert sol.decodeString("3[a]2[bc]") == "aaabcbc"
    assert sol.decodeString("3[a2[c]]") == "accaccacc"
    assert sol.decodeString("2[abc]3[cd]ef") == "abcabccdcdcdef"
    assert sol.decodeString("10[a]") == "aaaaaaaaaa"
    assert sol.decodeString("abc") == "abc"


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


