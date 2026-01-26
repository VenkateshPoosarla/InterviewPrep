"""187. Repeated DNA Sequences

Link: https://leetcode.com/problems/repeated-dna-sequences/

Pattern: Rolling hash / bit-encoding.

Encode each DNA char into 2 bits:
  A=00, C=01, G=10, T=11
Maintain a 20-bit rolling window for length 10 substrings.
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        L = 10
        if len(s) < L:
            return []

        enc = {"A": 0, "C": 1, "G": 2, "T": 3}
        mask = (1 << (2 * L)) - 1  # keep last 20 bits
        x = 0

        for i in range(L):
            x = (x << 2) | enc[s[i]]

        seen = {x}
        repeated = set()

        for i in range(L, len(s)):
            x = ((x << 2) | enc[s[i]]) & mask
            if x in seen:
                repeated.add(s[i - L + 1 : i + 1])
            else:
                seen.add(x)

        return list(repeated)


def run_tests() -> None:
    sol = Solution()
    out = sol.findRepeatedDnaSequences("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT")
    assert sorted(out) == sorted(["AAAAACCCCC", "CCCCCAAAAA"])
    assert sol.findRepeatedDnaSequences("AAAAAAAAAAAAA") == ["AAAAAAAAAA"]
    assert sol.findRepeatedDnaSequences("ACGT") == []


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


