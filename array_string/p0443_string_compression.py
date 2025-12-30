"""443. String Compression

Link: https://leetcode.com/problems/string-compression/

We must modify chars in-place and return the new length.

Two pointers:
- read pointer scans groups of same char
- write pointer writes compressed output

Example:
    a a a b b c c c
    write: a 3 b 2 c 3
"""

from __future__ import annotations

from typing import List


class Solution:
    def compress(self, chars: List[str]) -> int:
        n = len(chars)
        write = 0
        read = 0

        while read < n:
            ch = chars[read]
            start = read
            while read < n and chars[read] == ch:
                read += 1
            count = read - start

            chars[write] = ch
            write += 1

            if count > 1:
                for digit in str(count):
                    chars[write] = digit
                    write += 1

        return write


def run_tests() -> None:
    sol = Solution()

    chars = ["a", "a", "b", "b", "c", "c", "c"]
    k = sol.compress(chars)
    assert k == 6
    assert chars[:k] == ["a", "2", "b", "2", "c", "3"]

    chars = ["a"]
    k = sol.compress(chars)
    assert k == 1
    assert chars[:k] == ["a"]

    chars = ["a"] * 12
    k = sol.compress(chars)
    assert k == 3
    assert chars[:k] == ["a", "1", "2"]

    chars = ["a", "b", "c"]
    k = sol.compress(chars)
    assert k == 3
    assert chars[:k] == ["a", "b", "c"]


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


