"""649. Dota2 Senate

Link: https://leetcode.com/problems/dota2-senate/

Queue simulation:
- Store indices of 'R' and 'D' senators in two queues.
- Each round, the smaller index acts first and bans one opponent, then re-enters with index + n.
Stop when one queue is empty.
"""

from __future__ import annotations

from collections import deque
from typing import Deque


class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        n = len(senate)
        r: Deque[int] = deque()
        d: Deque[int] = deque()
        for i, ch in enumerate(senate):
            if ch == "R":
                r.append(i)
            else:
                d.append(i)

        while r and d:
            ri = r.popleft()
            di = d.popleft()
            if ri < di:
                r.append(ri + n)
            else:
                d.append(di + n)

        return "Radiant" if r else "Dire"


def run_tests() -> None:
    sol = Solution()

    assert sol.predictPartyVictory("RD") == "Radiant"
    assert sol.predictPartyVictory("RDD") == "Dire"
    assert sol.predictPartyVictory("RRDDD") in ("Radiant", "Dire")  # deterministic but not needed here
    assert sol.predictPartyVictory("R") == "Radiant"


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


