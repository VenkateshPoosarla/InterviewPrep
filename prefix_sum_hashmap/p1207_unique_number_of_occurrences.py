"""1207. Unique Number of Occurrences

Link: https://leetcode.com/problems/unique-number-of-occurrences/

Count frequencies, then ensure the set of counts has same size as list of counts.
"""

from __future__ import annotations

from collections import Counter
from typing import List


class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        freq = Counter(arr)
        counts = list(freq.values())
        return len(counts) == len(set(counts))


def run_tests() -> None:
    sol = Solution()

    assert sol.uniqueOccurrences([1, 2, 2, 1, 1, 3]) is True
    assert sol.uniqueOccurrences([1, 2]) is False
    assert sol.uniqueOccurrences([-3, 0, 1, -3, 1, 1, 1, -3, 10, 0]) is True
    assert sol.uniqueOccurrences([]) is True


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


