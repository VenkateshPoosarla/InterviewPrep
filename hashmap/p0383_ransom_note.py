"""383. Ransom Note

Link: https://leetcode.com/problems/ransom-note/

Problem:
Return True if `ransomNote` can be constructed using letters from `magazine`.
Each letter from magazine can only be used once.

Approach (count characters):
Count magazine letters; decrement for each letter needed by ransomNote.
If any needed count goes negative, impossible.

Complexity:
- Time: O(n + m)
- Space: O(1) (bounded by alphabet size)
"""

from __future__ import annotations

import sys
from collections import Counter


class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        counts = Counter(magazine)
        for ch in ransomNote:
            counts[ch] -= 1
            if counts[ch] < 0:
                return False
        return True


def run_tests() -> None:
    sol = Solution()
    assert sol.canConstruct("a", "b") is False
    assert sol.canConstruct("aa", "ab") is False
    assert sol.canConstruct("aa", "aab") is True


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
