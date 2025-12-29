"""71. Simplify Path

Link: https://leetcode.com/problems/simplify-path/

Problem:
Given an absolute Unix-style file path, simplify it:
- "." means current directory (ignore)
- ".." means go up one directory (pop)
- Multiple slashes are treated as a single slash
Return the canonical simplified path.

Approach (stack of path components):
Split by '/', process tokens:
- "" or "." -> skip
- ".." -> pop if possible
- else -> push directory name

Complexity:
- Time: O(n)
- Space: O(n)
"""

from __future__ import annotations

import sys


class Solution:
    def simplifyPath(self, path: str) -> str:
        stack: list[str] = []
        for part in path.split("/"):
            if part == "" or part == ".":
                continue
            if part == "..":
                if stack:
                    stack.pop()
            else:
                stack.append(part)
        return "/" + "/".join(stack)


def run_tests() -> None:
    sol = Solution()
    assert sol.simplifyPath("/home/") == "/home"
    assert sol.simplifyPath("/../") == "/"
    assert sol.simplifyPath("/home//foo/") == "/home/foo"
    assert sol.simplifyPath("/a/./b/../../c/") == "/c"
    assert sol.simplifyPath("/a/../../b/../c//.//") == "/c"
    assert sol.simplifyPath("/a//b////c/d//././/..") == "/a/b/c"


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
