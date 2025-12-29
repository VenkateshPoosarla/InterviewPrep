from __future__ import annotations

"""
Sync `top_interview_150/` structure + stubs (similar to `leetcode75/`).

What it does:
- Uses a cached copy of `https://leetcode-top-interview-150.github.io/` (stored as
  `top_interview_150_source.html`) as the canonical Top Interview 150 list.
- Scans this repo for existing `p####_*.py` files and DOES NOT duplicate problems that
  already exist (by problem number).
- Generates missing files as lightweight stubs with `run_tests()` and `sys.exit(0)`.

Run:
  python sync_top_interview_150.py

One-time cache (network):
  curl -k -L -s 'https://leetcode-top-interview-150.github.io/' > top_interview_150_source.html
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class Problem:
    section: str
    pid: int
    title: str


SECTION_TO_FOLDER: Dict[str, str] = {
    "Array/String": "array_string",
    "Two Pointers": "two_pointers",
    "Sliding Window": "sliding_window",
    "Matrix": "matrix",
    "Hashmap": "hashmap",
    "Intervals": "intervals",
    "Stack": "stack",
    "Linked List": "linked_list",
    "Binary Tree General": "binary_tree_general",
    "Binary Tree BFS": "binary_tree_bfs",
    "Binary Search Tree": "binary_search_tree",
    "Graph General": "graph_general",
    "Graph BFS": "graph_bfs",
    "Trie": "trie",
    "Backtracking": "backtracking",
    "Divide and Conquer": "divide_and_conquer",
    "Kadane’s Algorithm": "kadanes_algorithm",
    "Binary Search": "binary_search",
    "Heap": "heap",
    "Bit Manipulation": "bit_manipulation",
    "Math": "math",
    "1D DP": "dp_1d",
    "Multidimensional DP": "dp_multidim",
}


def _snake_case_title(title: str) -> str:
    t = title.lower()
    t = t.replace("’", "'").replace("–", "-").replace("—", "-")
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    t = re.sub(r"\s+", "_", t)
    return t or "problem"


def _padded(pid: int) -> str:
    return f"{pid:04d}"


def _ensure_package_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    init_py = path / "__init__.py"
    if not init_py.exists():
        init_py.write_text("", encoding="utf-8")


def _scan_existing_problem_files(repo_root: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    pat = re.compile(r"^p(\d{4})_.*\.py$")
    for p in repo_root.rglob("p[0-9][0-9][0-9][0-9]_*.py"):
        m = pat.match(p.name)
        if m:
            out.setdefault(int(m.group(1)), p)
    return out


def _load_top_interview_150(cached_html: Path) -> List[Problem]:
    text = cached_html.read_text(encoding="utf-8", errors="ignore")
    start = text.find('<h3 id="top-interview-150">')
    end = text.find('<h3 id="data-structure-i">', start)
    if start == -1 or end == -1:
        raise RuntimeError("Could not locate Top Interview 150 section in cached HTML.")
    chunk = text[start:end]

    sec_iter = list(
        re.finditer(r'<h4 id="top-interview-150-[^"]+">([^<]+)</h4>', chunk)
    )
    if not sec_iter:
        raise RuntimeError("No Top Interview 150 sections found in cached HTML.")

    problems: List[Problem] = []
    for i, m in enumerate(sec_iter):
        sec_title_full = m.group(1).strip()
        sec = sec_title_full.replace("Top Interview 150 ", "")
        s_start = m.end()
        s_end = sec_iter[i + 1].start() if i + 1 < len(sec_iter) else len(chunk)
        sec_html = chunk[s_start:s_end]

        rows = re.findall(
            r"<tr>\s*<td>(\d+)</td>\s*<td>(.*?)</td>", sec_html, flags=re.S
        )
        for num, title_html in rows:
            title = re.sub(r"<[^>]+>", "", title_html).strip()
            problems.append(Problem(section=sec, pid=int(num), title=title))

    if len({p.pid for p in problems}) != 150:
        raise RuntimeError("Expected 150 unique problems; cached HTML may have changed.")
    return problems


def _stub_file_contents(pid: int, title: str) -> str:
    return f'''""\"{pid}. {title}

Search: https://leetcode.com/problemset/?search={title.replace(" ", "%20")}

Note:
- This file is a stub generated to match the repo structure.
- Replace `solve()` with the LeetCode method signature when you implement it.
""\"

from __future__ import annotations

import sys
from typing import Any


class Solution:
    def solve(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def run_tests() -> None:
    sol = Solution()
    assert callable(getattr(sol, "solve"))


if __name__ == "__main__":
    run_tests()
    sys.exit(0)
'''


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    cached_html = repo_root / "top_interview_150_source.html"
    out_root = repo_root / "top_interview_150"

    if not cached_html.exists():
        print(
            "Missing `top_interview_150_source.html`.\n"
            "Create it once (network):\n"
            "  curl -k -L -s 'https://leetcode-top-interview-150.github.io/' > top_interview_150_source.html",
            file=sys.stderr,
        )
        return 2

    problems = _load_top_interview_150(cached_html)
    existing = _scan_existing_problem_files(repo_root)

    _ensure_package_dir(out_root)

    created = 0
    skipped_existing = 0
    for p in problems:
        folder = SECTION_TO_FOLDER[p.section]
        target_dir = out_root / folder
        fname = f"p{_padded(p.pid)}_{_snake_case_title(p.title)}.py"
        target_path = target_dir / fname

        if p.pid in existing:
            skipped_existing += 1
            continue

        _ensure_package_dir(target_dir)
        if not target_path.exists():
            target_path.write_text(_stub_file_contents(p.pid, p.title), encoding="utf-8")
            created += 1

    print(f"Top Interview 150: created {created} new stub files.")
    print(f"Top Interview 150: skipped {skipped_existing} problems already present in repo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


