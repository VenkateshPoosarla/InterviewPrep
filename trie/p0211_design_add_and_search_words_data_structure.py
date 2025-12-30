"""211. Design Add and Search Words Data Structure

Link: https://leetcode.com/problems/design-add-and-search-words-data-structure/

Problem:
Implement a data structure that supports:
- addWord(word): add a word
- search(word): return True if any previously added word matches `word`, where '.'
  can match any single character.

Approach (Trie + DFS for wildcard):
Store words in a trie. For search:
- If the current character is a letter, follow that edge.
- If it's '.', try all child edges recursively.

Complexity:
- addWord: O(L)
- search: O(L) typical; worst O(26^L) with many '.' and dense trie
Space: O(total characters added)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    end: bool = False


class WordDictionary:
    def __init__(self) -> None:
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.end = True

    def search(self, word: str) -> bool:
        def dfs(i: int, node: TrieNode) -> bool:
            if i == len(word):
                return node.end
            ch = word[i]
            if ch == ".":
                return any(dfs(i + 1, nxt) for nxt in node.children.values())
            if ch not in node.children:
                return False
            return dfs(i + 1, node.children[ch])

        return dfs(0, self.root)


def run_tests() -> None:
    wd = WordDictionary()
    wd.addWord("bad")
    wd.addWord("dad")
    wd.addWord("mad")
    assert wd.search("pad") is False
    assert wd.search("bad") is True
    assert wd.search(".ad") is True
    assert wd.search("b..") is True


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
