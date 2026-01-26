"""3. Longest Substring Without Repeating Characters

Link: https://leetcode.com/problems/longest-substring-without-repeating-characters/

Problem:
Return the length of the longest substring without repeating characters.

Approach (sliding window + last seen index):
Maintain a window [l..r] with unique chars.
Keep `last[c]` = last index where character c appeared.
When we see c at r:
- If last[c] >= l, it repeats inside the window -> move l = last[c] + 1
- Update best = max(best, r-l+1)

Complexity:
- Time: O(n)
- Space: O(k) where k is charset size (<= number of distinct chars in s)
"""

from __future__ import annotations

import sys


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        last: dict[str, int] = {}
        best = 0
        l = 0
        for r, ch in enumerate(s):
            if ch in last and last[ch] >= l:
                l = last[ch] + 1
            last[ch] = r
            best = max(best, r - l + 1)
        return best


def run_tests() -> None:
    sol = Solution()
    assert sol.lengthOfLongestSubstring("abcabcbb") == 3
    assert sol.lengthOfLongestSubstring("bbbbb") == 1
    assert sol.lengthOfLongestSubstring("pwwkew") == 3
    assert sol.lengthOfLongestSubstring("") == 0
    assert sol.lengthOfLongestSubstring(" ") == 1


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
