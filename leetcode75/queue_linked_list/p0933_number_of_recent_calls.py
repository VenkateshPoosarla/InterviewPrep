"""933. Number of Recent Calls

Link: https://leetcode.com/problems/number-of-recent-calls/

Maintain a queue of ping times; pop those older than (t-3000).
"""

from __future__ import annotations

from collections import deque
from typing import Deque


class RecentCounter:
    def __init__(self) -> None:
        self.q: Deque[int] = deque()

    def ping(self, t: int) -> int:
        self.q.append(t)
        threshold = t - 3000
        while self.q and self.q[0] < threshold:
            self.q.popleft()
        return len(self.q)


def run_tests() -> None:
    rc = RecentCounter()
    assert rc.ping(1) == 1
    assert rc.ping(100) == 2
    assert rc.ping(3001) == 3
    assert rc.ping(3002) == 3


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


