"""146. LRU Cache

Link: https://leetcode.com/problems/lru-cache/

Problem:
Design a data structure that follows the constraints of a Least Recently Used (LRU) cache:
- get(key): return value if key exists, else -1; mark as most recently used
- put(key, value): insert/update; if capacity exceeded, evict least recently used key

Approach (OrderedDict):
Python's `collections.OrderedDict` maintains insertion order and supports moving keys
to the end in O(1). We'll treat the end as "most recently used".

Operations:
- get: if present, move_to_end(key) and return value
- put: if key exists, update and move_to_end
       else insert; if over capacity, popitem(last=False) removes LRU

Complexity:
- Time: O(1) average per operation
- Space: O(capacity)
"""

from __future__ import annotations

import sys
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.od: "OrderedDict[int, int]" = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.od:
            return -1
        self.od.move_to_end(key)
        return self.od[key]

    def put(self, key: int, value: int) -> None:
        if key in self.od:
            self.od[key] = value
            self.od.move_to_end(key)
            return
        self.od[key] = value
        if len(self.od) > self.capacity:
            self.od.popitem(last=False)


def run_tests() -> None:
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1
    cache.put(3, 3)  # evicts key 2
    assert cache.get(2) == -1
    cache.put(4, 4)  # evicts key 1
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
