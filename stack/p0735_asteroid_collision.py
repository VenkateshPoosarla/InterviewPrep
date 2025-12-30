"""735. Asteroid Collision

Link: https://leetcode.com/problems/asteroid-collision/

Use a stack of "surviving asteroids" from left to right.
Only potential collision is when:
  stack top is moving right (positive)
  current asteroid is moving left (negative)

Resolve collisions until stable.

Visual:
  [5, 10, -5]
   st=[5]
   st=[5,10]
   cur=-5 collides with 10 (10 survives) => st=[5,10]
"""

from __future__ import annotations

from typing import List


class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        st: list[int] = []

        for a in asteroids:
            alive = True
            while alive and a < 0 and st and st[-1] > 0:
                top = st[-1]
                if top < -a:
                    st.pop()  # top explodes; keep checking with new top
                    continue
                if top == -a:
                    st.pop()  # both explode
                alive = False  # current a explodes (or both did)

            if alive:
                st.append(a)

        return st


def run_tests() -> None:
    sol = Solution()

    assert sol.asteroidCollision([5, 10, -5]) == [5, 10]
    assert sol.asteroidCollision([8, -8]) == []
    assert sol.asteroidCollision([10, 2, -5]) == [10]
    assert sol.asteroidCollision([-2, -1, 1, 2]) == [-2, -1, 1, 2]
    assert sol.asteroidCollision([]) == []


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


