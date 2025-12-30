"""1268. Search Suggestions System

Link: https://leetcode.com/problems/search-suggestions-system/

For each prefix of searchWord, return up to 3 lexicographically smallest products
that start with that prefix.

Efficient approach:
- sort products
- for each prefix, binary search (bisect_left) for insertion point of prefix
- take next up to 3 items and filter by startswith(prefix)
"""

from __future__ import annotations

import bisect
from typing import List


class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        products = sorted(products)
        out: List[List[str]] = []
        prefix = ""

        for ch in searchWord:
            prefix += ch
            i = bisect.bisect_left(products, prefix)
            suggestions: list[str] = []
            for j in range(i, min(i + 3, len(products))):
                if products[j].startswith(prefix):
                    suggestions.append(products[j])
                else:
                    break
            out.append(suggestions)

        return out


def run_tests() -> None:
    sol = Solution()

    products = ["mobile", "mouse", "moneypot", "monitor", "mousepad"]
    assert sol.suggestedProducts(products, "mouse") == [
        ["mobile", "moneypot", "monitor"],
        ["mobile", "moneypot", "monitor"],
        ["mouse", "mousepad"],
        ["mouse", "mousepad"],
        ["mouse", "mousepad"],
    ]

    assert sol.suggestedProducts(["havana"], "havana") == [["havana"]] * 6
    assert sol.suggestedProducts(["bags", "baggage", "banner", "box", "cloths"], "bags") == [
        ["baggage", "bags", "banner"],
        ["baggage", "bags", "banner"],
        ["baggage", "bags"],
        ["bags"],
    ]


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


