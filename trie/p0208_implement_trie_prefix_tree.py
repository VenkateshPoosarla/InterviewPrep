"""208. Implement Trie (Prefix Tree)

Link: https://leetcode.com/problems/implement-trie-prefix-tree/

Trie nodes store:
- children: mapping char -> node
- is_end: whether a word ends here
"""

from __future__ import annotations


class TrieNode:
    __slots__ = ("children", "is_end")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_end: bool = False


class Trie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self._walk(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        return self._walk(prefix) is not None

    def _walk(self, s: str) -> TrieNode | None:
        node = self.root
        for ch in s:
            nxt = node.children.get(ch)
            if nxt is None:
                return None
            node = nxt
        return node


def run_tests() -> None:
    trie = Trie()
    trie.insert("apple")
    assert trie.search("apple") is True
    assert trie.search("app") is False
    assert trie.startsWith("app") is True
    trie.insert("app")
    assert trie.search("app") is True
    assert trie.startsWith("") is True


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


