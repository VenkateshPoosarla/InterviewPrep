"""380. Insert Delete GetRandom O(1)

Link: https://leetcode.com/problems/insert-delete-getrandom-o1/

Problem:
Design a data structure that supports insert, remove, and getRandom in average O(1).

Approach:
Use:
- A list `vals` to store the elements (so we can pick random by index).
- A dict `pos` mapping value -> index in `vals` (so we can find/remove in O(1)).

Removal trick:
To remove vals[idx] in O(1), swap it with the last element and pop:
  vals[idx] = last
  pos[last] = idx
  pop last

Complexity (average):
- insert: O(1)
- remove: O(1)
- getRandom: O(1)
Space: O(n)
"""

from __future__ import annotations

import random
import sys


class RandomizedSet:
    def __init__(self) -> None:
        self.vals: list[int] = []
        self.pos: dict[int, int] = {}

    def insert(self, val: int) -> bool:
        if val in self.pos:
            return False
        self.pos[val] = len(self.vals)
        self.vals.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.pos:
            return False

        idx = self.pos[val]
        last = self.vals[-1]

        # Move last into the removed slot (unless we're removing the last itself).
        self.vals[idx] = last
        self.pos[last] = idx

        # Remove the last slot and delete mapping.
        self.vals.pop()
        del self.pos[val]
        return True

    def getRandom(self) -> int:
        return random.choice(self.vals)


def run_tests() -> None:
    random.seed(0)
    rs = RandomizedSet()

    assert rs.insert(1) is True
    assert rs.insert(1) is False
    assert rs.insert(2) is True

    # Random must be one of existing elements.
    x = rs.getRandom()
    assert x in {1, 2}

    assert rs.remove(1) is True
    assert rs.remove(1) is False
    assert rs.vals == [2]
    assert rs.getRandom() == 2


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
