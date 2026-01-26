"""307. Range Sum Query - Mutable

Link: https://leetcode.com/problems/range-sum-query-mutable/

Pattern: Fenwick Tree (Binary Indexed Tree) for:
- point update
- prefix sum / range sum
"""

from __future__ import annotations

import sys
from typing import List


class Fenwick:
    def __init__(self, n: int) -> None:
        self.n = n
        self.bit = [0] * (n + 1)

    def add(self, i: int, delta: int) -> None:
        i += 1
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def sum_prefix(self, i: int) -> int:
        # sum of [0..i)
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s

    def sum_range(self, l: int, r: int) -> int:
        # sum of [l..r] inclusive
        return self.sum_prefix(r + 1) - self.sum_prefix(l)


class NumArray:
    def __init__(self, nums: List[int]):
        self.arr = [int(x) for x in nums]
        self.fw = Fenwick(len(self.arr))
        for i, v in enumerate(self.arr):
            self.fw.add(i, v)

    def update(self, index: int, val: int) -> None:
        val = int(val)
        delta = val - self.arr[index]
        self.arr[index] = val
        self.fw.add(index, delta)

    def sumRange(self, left: int, right: int) -> int:
        return self.fw.sum_range(left, right)


def run_tests() -> None:
    na = NumArray([1, 3, 5])
    assert na.sumRange(0, 2) == 9
    na.update(1, 2)
    assert na.sumRange(0, 2) == 8
    assert na.sumRange(1, 1) == 2


if __name__ == "__main__":
    run_tests()
    sys.exit(0)


