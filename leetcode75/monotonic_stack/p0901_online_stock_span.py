"""901. Online Stock Span

Link: https://leetcode.com/problems/online-stock-span/

Maintain a monotonic decreasing stack of (price, span).
When a new price arrives, pop while top.price <= price and accumulate spans.
The accumulated span is the answer for this price.
"""

from __future__ import annotations


class StockSpanner:
    def __init__(self) -> None:
        self.st: list[tuple[int, int]] = []  # (price, span)

    def next(self, price: int) -> int:
        span = 1
        while self.st and self.st[-1][0] <= price:
            _, s = self.st.pop()
            span += s
        self.st.append((price, span))
        return span


def run_tests() -> None:
    s = StockSpanner()
    assert s.next(100) == 1
    assert s.next(80) == 1
    assert s.next(60) == 1
    assert s.next(70) == 2
    assert s.next(60) == 1
    assert s.next(75) == 4
    assert s.next(85) == 6


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


