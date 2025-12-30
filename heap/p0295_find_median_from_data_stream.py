"""295. Find Median from Data Stream

Link: https://leetcode.com/problems/find-median-from-data-stream/

Problem:
Design a data structure that supports adding numbers and finding the median.

Approach (two heaps):
Maintain:
- `low` as a max-heap (store negatives) for the lower half
- `high` as a min-heap for the upper half
Balance sizes so that len(low) == len(high) or len(low) == len(high)+1.
Median:
- If odd: top of low
- If even: average of tops

Complexity:
- addNum: O(log n)
- findMedian: O(1)
Space: O(n)
"""

from __future__ import annotations

import heapq
import sys


class MedianFinder:
    def __init__(self) -> None:
        self.low: list[int] = []   # max-heap via negatives
        self.high: list[int] = []  # min-heap

    def addNum(self, num: int) -> None:
        heapq.heappush(self.low, -num)
        # Ensure ordering: max(low) <= min(high)
        if self.high and -self.low[0] > self.high[0]:
            x = -heapq.heappop(self.low)
            heapq.heappush(self.high, x)
        # Balance sizes
        if len(self.low) > len(self.high) + 1:
            heapq.heappush(self.high, -heapq.heappop(self.low))
        elif len(self.high) > len(self.low):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def findMedian(self) -> float:
        if len(self.low) > len(self.high):
            return float(-self.low[0])
        return (-self.low[0] + self.high[0]) / 2.0


def run_tests() -> None:
    mf = MedianFinder()
    mf.addNum(1)
    mf.addNum(2)
    assert mf.findMedian() == 1.5
    mf.addNum(3)
    assert mf.findMedian() == 2.0


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
