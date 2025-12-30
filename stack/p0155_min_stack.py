"""155. Min Stack

Link: https://leetcode.com/problems/min-stack/

Problem:
Design a stack that supports push, pop, top, and retrieving the minimum element in O(1).

Approach (value stack + min stack):
Maintain:
- stack: pushed values
- mins: mins[i] is the minimum of stack[0..i]
On push(x), mins.push(min(x, mins[-1])).
On pop(), pop both.

Complexity:
- Time: O(1) per operation
- Space: O(n)
"""

from __future__ import annotations

import sys


class MinStack:
    def __init__(self) -> None:
        self.stack: list[int] = []
        self.mins: list[int] = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.mins:
            self.mins.append(val)
        else:
            self.mins.append(min(val, self.mins[-1]))

    def pop(self) -> None:
        self.stack.pop()
        self.mins.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.mins[-1]


def run_tests() -> None:
    ms = MinStack()
    ms.push(-2)
    ms.push(0)
    ms.push(-3)
    assert ms.getMin() == -3
    ms.pop()
    assert ms.top() == 0
    assert ms.getMin() == -2


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
