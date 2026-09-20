"""Pytest configuration.

``tmp_path`` is only as safe as the temp root pytest resolves. Some sandboxed
runners hand Python the working directory as that root, so
``tempfile.gettempdir()`` returns the repository itself and pytest builds
``pytest-of-<user>/`` inside the working tree. The tests are not at fault --
they all take ``tmp_path`` and never name a relative path -- so the fix belongs
here: pin the temp root once, before any fixture reads it, and refuse to run
rather than scatter scratch files through the repo.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()


def _is_inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return False
    return True


def _candidate_temp_roots() -> list[Path]:
    """Temp roots to try, most explicit first."""
    named = [os.environ.get(name) for name in ("TMPDIR", "TEMP", "TMP")]
    candidates = [Path(value) for value in named if value]
    if os.name == "nt":
        candidates.append(Path.home() / "AppData" / "Local" / "Temp")
    else:
        candidates.append(Path("/tmp"))
    return candidates


def _pin_temp_root() -> None:
    """Point pytest and ``tempfile`` at a writable temp root outside the repo."""
    if os.environ.get("PYTEST_DEBUG_TEMPROOT"):
        return  # An explicit override is the caller's business, not ours.

    current = Path(tempfile.gettempdir())
    if not _is_inside_repo(current):
        return  # Already sane. Leave the environment alone.

    for candidate in _candidate_temp_roots():
        if _is_inside_repo(candidate) or not candidate.is_dir():
            continue
        resolved = str(candidate.resolve())
        os.environ["PYTEST_DEBUG_TEMPROOT"] = resolved
        tempfile.tempdir = resolved  # Also covers mkdtemp() under test.
        return

    raise RuntimeError(
        f"tempfile.gettempdir() resolves to {current}, which is inside the "
        f"repository at {REPO_ROOT}, and no candidate in TMPDIR/TEMP/TMP is a "
        "usable alternative. Running would write test scratch files into the "
        "working tree. Set TMPDIR to a writable directory outside the repo."
    )


_pin_temp_root()
