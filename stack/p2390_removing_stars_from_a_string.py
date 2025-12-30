"""2390. Removing Stars From a String

Link: https://leetcode.com/problems/removing-stars-from-a-string/

Process left-to-right:
- If char != '*', push onto stack
- If char == '*', pop one previous character (guaranteed possible by constraints)

Visual:
  s = "leet**cod*e"
      stack: l e e t
      * -> pop t
      * -> pop e
      then c o d
      * -> pop d
      then e
  result: "lecoe"
"""

from __future__ import annotations


class Solution:
    def removeStars(self, s: str) -> str:
        st: list[str] = []
        for ch in s:
            if ch == "*":
                st.pop()
            else:
                st.append(ch)
        return "".join(st)


def run_tests() -> None:
    sol = Solution()

    assert sol.removeStars("leet**cod*e") == "lecoe"
    assert sol.removeStars("erase*****") == ""
    assert sol.removeStars("a*b*c*") == ""
    assert sol.removeStars("abc") == "abc"


if __name__ == "__main__":
    run_tests()
    import sys

    sys.exit(0)


