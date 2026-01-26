"""49. Group Anagrams

Link: https://leetcode.com/problems/group-anagrams/

Problem:
Group strings that are anagrams of each other.

Approach:
Anagrams share the same sorted-character signature.
Use a dict: signature -> list of words.

Complexity:
- Time: O(n * k log k) where k is average word length (sorting each word)
- Space: O(n * k)
"""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import DefaultDict, List


class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        groups: DefaultDict[str, List[str]] = defaultdict(list)
        for w in strs:
            key = "".join(sorted(w))
            groups[key].append(w)
        return list(groups.values())


def run_tests() -> None:
    sol = Solution()
    out = sol.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    normalized = sorted([sorted(g) for g in out])
    assert normalized == sorted([["ate", "eat", "tea"], ["bat"], ["nat", "tan"]])


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
