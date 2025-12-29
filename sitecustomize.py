"""
Repo-wide Python startup customization.

Why this exists:
- Some generated stub problem files call `sys.exit(0)` without importing `sys`.
- When you run scripts from the repo root (the common workflow here), Python will
  automatically import `sitecustomize` (via `site`) if it is on `sys.path`.

This keeps the repo runnable while you gradually replace stubs with real solutions.
"""

from __future__ import annotations

import builtins
import sys as _sys

if not hasattr(builtins, "sys"):
    builtins.sys = _sys


