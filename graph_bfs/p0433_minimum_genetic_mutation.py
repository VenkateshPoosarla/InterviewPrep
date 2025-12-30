"""433. Minimum Genetic Mutation

Link: https://leetcode.com/problems/minimum-genetic-mutation/

Problem:
Given startGene, endGene, and a bank of valid genes, return the minimum number of
single-character mutations to reach endGene from startGene, where each intermediate gene
must be in bank. If impossible, return -1.

Approach (BFS):
Each gene is a node; edges connect genes that differ by exactly one character.
BFS from startGene to find shortest path to endGene.
To generate neighbors efficiently, for each position try replacing with {A,C,G,T}.

Complexity:
- Time: O(|bank| * L * 4) in worst case (L=8)
- Space: O(|bank|)
"""

from __future__ import annotations

import sys
from collections import deque
from typing import Deque, List, Set


class Solution:
    def minMutation(self, startGene: str, endGene: str, bank: List[str]) -> int:
        bank_set: Set[str] = set(bank)
        if endGene not in bank_set:
            return -1

        genes = ["A", "C", "G", "T"]
        q: Deque[tuple[str, int]] = deque([(startGene, 0)])
        seen = {startGene}

        while q:
            cur, steps = q.popleft()
            if cur == endGene:
                return steps
            arr = list(cur)
            for i in range(len(arr)):
                old = arr[i]
                for g in genes:
                    if g == old:
                        continue
                    arr[i] = g
                    nxt = "".join(arr)
                    if nxt in bank_set and nxt not in seen:
                        seen.add(nxt)
                        q.append((nxt, steps + 1))
                arr[i] = old

        return -1


def run_tests() -> None:
    sol = Solution()
    assert sol.minMutation("AACCGGTT", "AACCGGTA", ["AACCGGTA"]) == 1
    assert sol.minMutation("AACCGGTT", "AAACGGTA", ["AACCGGTA", "AACCGCTA", "AAACGGTA"]) == 2
    assert sol.minMutation("AAAAACCC", "AACCCCCC", ["AAAACCCC", "AAACCCCC", "AACCCCCC"]) == 3
    assert sol.minMutation("AACCGGTT", "AACCGGTA", []) == -1


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
