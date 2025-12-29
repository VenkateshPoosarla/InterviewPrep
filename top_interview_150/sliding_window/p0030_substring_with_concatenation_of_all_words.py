"""30. Substring with Concatenation of All Words

Link: https://leetcode.com/problems/substring-with-concatenation-of-all-words/

Problem:
Given a string `s` and a list of words `words` (all same length), return all starting
indices of substrings in `s` that are a concatenation of each word exactly once, in any order.

Approach (sliding window by word length):
Let word_len = len(words[0]), total_len = word_len * len(words).
We consider `word_len` different offsets (0..word_len-1), and slide a window in steps of word_len.

Maintain:
- need[word] = required count from words
- window[word] = current count inside window
- count = number of words currently in window

When a word is overused, shrink from left until counts valid again.
When count == len(words), record left index.

Complexity:
- Time: O(|s|) windows, each word enters/leaves once per offset
- Space: O(#distinct words)
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from typing import DefaultDict, List


class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        if not s or not words or not words[0]:
            return []

        word_len = len(words[0])
        k = len(words)
        total_len = word_len * k
        if len(s) < total_len:
            return []

        need = Counter(words)
        res: List[int] = []

        for offset in range(word_len):
            left = offset
            window: DefaultDict[str, int] = defaultdict(int)
            count = 0

            for right in range(offset, len(s) - word_len + 1, word_len):
                w = s[right : right + word_len]
                if w not in need:
                    # Reset window after invalid word.
                    window.clear()
                    count = 0
                    left = right + word_len
                    continue

                window[w] += 1
                count += 1

                # If we have too many of w, shrink from left.
                while window[w] > need[w]:
                    left_word = s[left : left + word_len]
                    window[left_word] -= 1
                    count -= 1
                    left += word_len

                if count == k:
                    res.append(left)
                    # Move left by one word to look for the next match.
                    left_word = s[left : left + word_len]
                    window[left_word] -= 1
                    count -= 1
                    left += word_len

        return res


def run_tests() -> None:
    sol = Solution()

    out = sol.findSubstring("barfoothefoobarman", ["foo", "bar"])
    assert sorted(out) == [0, 9]

    out = sol.findSubstring("wordgoodgoodgoodbestword", ["word", "good", "best", "word"])
    assert out == []

    out = sol.findSubstring("barfoofoobarthefoobarman", ["bar", "foo", "the"])
    assert sorted(out) == [6, 9, 12]


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
