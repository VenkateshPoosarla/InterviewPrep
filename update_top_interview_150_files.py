from __future__ import annotations

"""
Bulk-update `top_interview_150/` files:
- Ensure each file has a LeetCode *problem page* link in this format:
    https://leetcode.com/problems/<slug>/
- Add a consistent header/comment scaffold
- Ensure `import sys` exists (files run tests then exit)

This is intended to be run from the repo root:
  python update_top_interview_150_files.py

It relies on `top_interview_150_source.html` (cached mirror HTML).
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProblemMeta:
    pid: int
    title: str
    slug: str


def _parse_source(html_path: Path) -> dict[int, ProblemMeta]:
    """
    Parse `top_interview_150_source.html` (from https://leetcode-top-interview-150.github.io/)
    and return pid -> (title, slug).

    The mirror encodes LeetCode links like: href="https://leetcode.com/problems/<slug>/"
    """
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    start = text.find('<h3 id="top-interview-150">')
    end = text.find('<h3 id="data-structure-i">', start)
    if start == -1 or end == -1:
        raise RuntimeError("Could not locate Top Interview 150 section in source HTML.")

    chunk = text[start:end]
    # rows look like:
    # <tr>
    #   <td>0088</td>
    #   <td><a href="https://leetcode.com/problems/merge-sorted-array/">Merge Sorted Array</a></td>
    rows = re.findall(
        r"<tr>\s*<td>(\d+)</td>\s*<td>\s*(?:<a[^>]*href=\"([^\"]+)\"[^>]*>)?([^<]+)(?:</a>)?\s*</td>",
        chunk,
        flags=re.S,
    )
    out: dict[int, ProblemMeta] = {}
    for num_s, href, title in rows:
        pid = int(num_s)
        title = title.strip()
        slug = ""
        if href:
            m = re.search(r"leetcode\.com/problems/([^/]+)/?", href)
            if m:
                slug = m.group(1)
        if not slug:
            # last resort: derive slug from title (LeetCode slugs match this for most problems)
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        out[pid] = ProblemMeta(pid=pid, title=title, slug=slug)

    if len(out) < 150:
        raise RuntimeError(f"Expected ~150 rows, got {len(out)}; source HTML may have changed.")
    return out


def _rewrite_file(path: Path, meta: ProblemMeta) -> bool:
    """
    Returns True if file changed.
    """
    original = path.read_text(encoding="utf-8", errors="ignore")

    # If file doesn't start with a docstring, we won't try to do fancy rewriting.
    if not original.lstrip().startswith('"""'):
        return False

    # Ensure sys import exists (before typing import is fine).
    if "\nimport sys\n" not in original and "\nimport sys\r\n" not in original:
        original = original.replace("from __future__ import annotations\n\n", "from __future__ import annotations\n\nimport sys\n\n")

    # Replace any "Search:" line with "Link:".
    leetcode_link = f"https://leetcode.com/problems/{meta.slug}/"
    updated = re.sub(
        r"(?m)^\s*(Search|Link)\s*:\s*https?://.*$",
        f"Link: {leetcode_link}",
        original,
    )

    # If there was no Search/Link line, insert Link after the title line in docstring.
    if "Link:" not in updated.splitlines()[0:15]:
        updated = re.sub(
            r'"""(\d+)\.\s*[^\n]+\n',
            lambda m: f'"""{m.group(1)}. {meta.title}\n\nLink: {leetcode_link}\n',
            updated,
            count=1,
        )

    # Add a small scaffold if missing (keep it minimal, don’t overwrite custom notes).
    if "Approach:" not in updated:
        updated = re.sub(
            r'"""([\s\S]*?)"""',
            lambda m: (
                '"""' + m.group(1).rstrip() + "\n\n"
                "Approach:\n"
                "- TODO\n\n"
                "Complexity:\n"
                "- Time: TODO\n"
                "- Space: TODO\n"
                '"""'
            ),
            updated,
            count=1,
        )

    if updated != path.read_text(encoding="utf-8", errors="ignore"):
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    src = repo_root / "top_interview_150_source.html"
    if not src.exists():
        print("Missing `top_interview_150_source.html` in repo root.", file=sys.stderr)
        return 2

    meta = _parse_source(src)
    top = repo_root / "top_interview_150"
    changed = 0
    scanned = 0

    for p in top.rglob("p[0-9][0-9][0-9][0-9]_*.py"):
        scanned += 1
        m = re.match(r"p(\d{4})_", p.name)
        if not m:
            continue
        pid = int(m.group(1))
        if pid not in meta:
            continue
        if _rewrite_file(p, meta[pid]):
            changed += 1

    print(f"Scanned {scanned} files, updated {changed}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


