"""135. Candy

Link: https://leetcode.com/problems/candy/

Problem:
Each child has a rating. Give candies such that:
- Each child has at least 1 candy
- Children with a higher rating than an adjacent child get more candies than that neighbor
Return the minimum number of candies needed.

Approach (two passes):
Start everyone at 1 candy.
1) Left-to-right: if rating[i] > rating[i-1], candies[i] = candies[i-1] + 1
2) Right-to-left: if rating[i] > rating[i+1], candies[i] = max(candies[i], candies[i+1] + 1)

This satisfies both neighbor constraints with minimal increments.

Complexity:
- Time: O(n)
- Space: O(n)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        if n == 0:
            return 0

        candies = [1] * n
        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                candies[i] = candies[i - 1] + 1

        for i in range(n - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                candies[i] = max(candies[i], candies[i + 1] + 1)

        return sum(candies)


def run_tests() -> None:
    sol = Solution()
    assert sol.candy([1, 0, 2]) == 5
    assert sol.candy([1, 2, 2]) == 4
    assert sol.candy([1]) == 1
    assert sol.candy([1, 3, 4, 5, 2]) == 11


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
