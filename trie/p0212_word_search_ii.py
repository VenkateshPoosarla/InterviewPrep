"""212. Word Search II

Link: https://leetcode.com/problems/word-search-ii/

Problem:
Given an m x n board of letters and a list of words, return all words present in the
board. Words are formed by sequentially adjacent cells (horizontal/vertical) and you
may not reuse the same cell in a single word.

Approach (Trie + DFS backtracking):
Build a trie of the words. Then start DFS from each cell:
- Walk the trie as we walk the board.
- When we reach a trie node that represents a complete word, record it.
- Mark a cell visited by temporarily replacing its character.
We also prune by deleting matched words from the trie node (set word=None).

Complexity:
- Time: depends on branching; trie pruning reduces redundant work significantly.
- Space: O(total word chars) for trie + O(path length) recursion.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    word: Optional[str] = None  # store full word at terminal nodes


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        if not board or not board[0] or not words:
            return []

        root = TrieNode()
        for w in words:
            node = root
            for ch in w:
                node = node.children.setdefault(ch, TrieNode())
            node.word = w

        m, n = len(board), len(board[0])
        res: List[str] = []

        def dfs(r: int, c: int, node: TrieNode) -> None:
            ch = board[r][c]
            if ch not in node.children:
                return
            nxt = node.children[ch]

            if nxt.word is not None:
                res.append(nxt.word)
                nxt.word = None  # avoid duplicates

            board[r][c] = "#"  # mark visited
            if r > 0 and board[r - 1][c] != "#":
                dfs(r - 1, c, nxt)
            if r + 1 < m and board[r + 1][c] != "#":
                dfs(r + 1, c, nxt)
            if c > 0 and board[r][c - 1] != "#":
                dfs(r, c - 1, nxt)
            if c + 1 < n and board[r][c + 1] != "#":
                dfs(r, c + 1, nxt)
            board[r][c] = ch  # unmark

            # prune: if nxt has no children and no word, remove it
            if not nxt.children and nxt.word is None:
                del node.children[ch]

        for r in range(m):
            for c in range(n):
                dfs(r, c, root)

        return res


def run_tests() -> None:
    sol = Solution()
    board = [
        ["o", "a", "a", "n"],
        ["e", "t", "a", "e"],
        ["i", "h", "k", "r"],
        ["i", "f", "l", "v"],
    ]
    out = sol.findWords(board, ["oath", "pea", "eat", "rain"])
    assert sorted(out) == ["eat", "oath"]

    board = [["a", "b"], ["c", "d"]]
    out = sol.findWords(board, ["abcb"])
    assert out == []


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
