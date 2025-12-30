"""460. LFU Cache

Link: https://leetcode.com/problems/lfu-cache/

Pattern: Design DS (frequency + recency).

Maintain:
- key -> (value, freq)
- freq -> OrderedDict of keys (acts as LRU within same freq)
- min_freq: smallest freq present in cache

All ops are O(1) amortized.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from typing import Dict, Tuple


class LFUCache:
    def __init__(self, capacity: int):
        self.cap = int(capacity)
        self.min_freq = 0
        self.kv: Dict[int, Tuple[int, int]] = {}  # key -> (value, freq)
        self.buckets: Dict[int, OrderedDict[int, None]] = {}  # freq -> keys in LRU order

    def _touch(self, key: int, new_value: int | None = None) -> None:
        val, f = self.kv[key]
        if new_value is not None:
            val = new_value

        # remove from old freq bucket
        od = self.buckets[f]
        od.pop(key, None)
        if not od:
            self.buckets.pop(f, None)
            if self.min_freq == f:
                self.min_freq += 1

        # add to new freq bucket
        nf = f + 1
        if nf not in self.buckets:
            self.buckets[nf] = OrderedDict()
        self.buckets[nf][key] = None
        self.kv[key] = (val, nf)

    def get(self, key: int) -> int:
        key = int(key)
        if key not in self.kv:
            return -1
        self._touch(key)
        return self.kv[key][0]

    def put(self, key: int, value: int) -> None:
        key = int(key)
        value = int(value)
        if self.cap <= 0:
            return

        if key in self.kv:
            self._touch(key, new_value=value)
            return

        if len(self.kv) >= self.cap:
            # evict LFU + LRU within bucket
            od = self.buckets[self.min_freq]
            evict_key, _ = od.popitem(last=False)
            if not od:
                self.buckets.pop(self.min_freq, None)
            self.kv.pop(evict_key, None)

        # insert with freq = 1
        self.kv[key] = (value, 1)
        if 1 not in self.buckets:
            self.buckets[1] = OrderedDict()
        self.buckets[1][key] = None
        self.min_freq = 1


def run_tests() -> None:
    # Example from prompt
    lfu = LFUCache(2)
    lfu.put(1, 1)
    lfu.put(2, 2)
    assert lfu.get(1) == 1
    lfu.put(3, 3)  # evicts key 2
    assert lfu.get(2) == -1
    assert lfu.get(3) == 3
    lfu.put(4, 4)  # evicts key 1
    assert lfu.get(1) == -1
    assert lfu.get(3) == 3
    assert lfu.get(4) == 4

    # capacity 0
    lfu = LFUCache(0)
    lfu.put(1, 1)
    assert lfu.get(1) == -1


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


