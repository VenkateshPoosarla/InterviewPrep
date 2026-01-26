"""76. Minimum Window Substring

Link: https://leetcode.com/problems/minimum-window-substring/

Problem:
Given strings `s` and `t`, return the minimum window substring of `s` such that every
character in `t` (including multiplicity) is included in the window. If none, return "".

Approach (sliding window with counts):
- need[c] = how many of c we still need to satisfy t
- missing = total number of characters still missing
Expand right pointer r:
  - decrement need[s[r]]; if it was >0, we satisfied one missing char
Once missing == 0, try to shrink from left while still valid, updating best window.

Complexity:
- Time: O(|s| + |t|)
- Space: O(|alphabet|)
"""

from __future__ import annotations

import sys
from collections import Counter


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not t or not s:
            return ""

        need = Counter(t)
        missing = len(t)

        best_len = float("inf")
        best_l = 0
        l = 0

        for r, ch in enumerate(s):
            if need[ch] > 0:
                missing -= 1
            need[ch] -= 1

            # If window covers t, shrink from left.
            while missing == 0:
                if r - l + 1 < best_len:
                    best_len = r - l + 1
                    best_l = l

                left_ch = s[l]
                need[left_ch] += 1
                if need[left_ch] > 0:
                    missing += 1
                l += 1

        if best_len == float("inf"):
            return ""
        return s[best_l : best_l + int(best_len)]


def run_tests() -> None:
    sol = Solution()
    assert sol.minWindow("ADOBECODEBANC", "ABC") == "BANC"
    assert sol.minWindow("a", "a") == "a"
    assert sol.minWindow("a", "aa") == ""
    assert sol.minWindow("aa", "aa") == "aa"


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
