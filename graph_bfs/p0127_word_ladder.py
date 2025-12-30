"""127. Word Ladder

Link: https://leetcode.com/problems/word-ladder/

Problem:
Given beginWord, endWord, and a wordList, return the length of the shortest transformation
sequence from beginWord to endWord such that:
- Only one letter can be changed at a time
- Each transformed word must exist in wordList
Return 0 if no such sequence exists.

Approach (BFS + wildcard buckets):
Two words are adjacent if they differ by one character.
Precompute buckets for word patterns where one position is replaced by '*':
  hot -> *ot, h*t, ho*
Words sharing a bucket are neighbors.
Then BFS from beginWord to find shortest path to endWord.

Complexity:
- Precompute: O(N * L)
- BFS: O(N * L) typical
Space: O(N * L)
"""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from typing import DefaultDict, Deque, List, Set


class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        words: Set[str] = set(wordList)
        if endWord not in words:
            return 0

        L = len(beginWord)
        buckets: DefaultDict[str, List[str]] = defaultdict(list)
        for w in words:
            for i in range(L):
                buckets[w[:i] + "*" + w[i + 1 :]].append(w)

        q: Deque[tuple[str, int]] = deque([(beginWord, 1)])
        seen = {beginWord}
        while q:
            w, dist = q.popleft()
            if w == endWord:
                return dist
            for i in range(L):
                pat = w[:i] + "*" + w[i + 1 :]
                for nei in buckets.get(pat, []):
                    if nei not in seen:
                        seen.add(nei)
                        q.append((nei, dist + 1))
                # optional: clear to reduce repeats
                buckets[pat] = []
        return 0


def run_tests() -> None:
    sol = Solution()
    assert sol.ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]) == 5
    assert sol.ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log"]) == 0
    assert sol.ladderLength("a", "c", ["a", "b", "c"]) == 2


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
