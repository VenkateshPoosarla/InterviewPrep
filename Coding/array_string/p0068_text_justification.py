"""68. Text Justification

Link: https://leetcode.com/problems/text-justification/

Problem:
Given a list of words and a max width, format the text so that each line has exactly
maxWidth characters and is fully (left and right) justified.

Rules:
- Pack as many words as possible per line.
- For non-last lines:
  - Distribute spaces as evenly as possible between words.
  - If extra spaces remain, put them in the leftmost gaps.
- For the last line (or a line with one word): left-justify and pad the end with spaces.

Approach:
Greedily choose words for each line (two pointers i..j).
Then build the line depending on whether it's the last line / single-word line.

Complexity:
- Time: O(total characters)
- Space: O(maxWidth) per line (output)
"""

from __future__ import annotations

import sys
from typing import List


class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        res: List[str] = []
        n = len(words)
        i = 0

        while i < n:
            # Greedily find the rightmost j we can include in this line.
            j = i
            letters = 0
            while j < n:
                # Needed width if we include words[j]:
                # letters + len(words[j]) + spaces_between_words
                next_letters = letters + len(words[j])
                gaps = j - i  # spaces between words if we include j
                if next_letters + gaps > maxWidth:
                    break
                letters = next_letters
                j += 1

            line_words = words[i:j]
            num_words = len(line_words)
            spaces_needed = maxWidth - letters

            is_last_line = j == n
            if is_last_line or num_words == 1:
                # Left-justify: single spaces between words, remainder at the end.
                line = " ".join(line_words)
                line += " " * (maxWidth - len(line))
                res.append(line)
            else:
                # Fully justify: distribute spaces across gaps.
                gaps = num_words - 1
                base = spaces_needed // gaps
                extra = spaces_needed % gaps  # leftmost gaps get one extra

                parts: List[str] = []
                for idx, w in enumerate(line_words):
                    parts.append(w)
                    if idx < gaps:
                        # base spaces + maybe one extra for leftmost gaps
                        spaces = base + (1 if idx < extra else 0)
                        parts.append(" " * spaces)
                res.append("".join(parts))

            i = j

        return res


def run_tests() -> None:
    sol = Solution()

    words = ["This", "is", "an", "example", "of", "text", "justification."]
    out = sol.fullJustify(words, 16)
    assert out == [
        "This    is    an",
        "example  of text",
        "justification.  ",
    ]

    words = ["What", "must", "be", "acknowledgment", "shall", "be"]
    out = sol.fullJustify(words, 16)
    assert out == [
        "What   must   be",
        "acknowledgment  ",
        "shall be        ",
    ]

    words = ["Science", "is", "what", "we", "understand", "well", "enough", "to", "explain", "to", "a", "computer.", "Art", "is", "everything", "else", "we", "do"]
    out = sol.fullJustify(words, 20)
    assert out[0] == "Science  is  what we"
    assert out[-1] == "else we do          "
    assert all(len(line) == 20 for line in out)


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
