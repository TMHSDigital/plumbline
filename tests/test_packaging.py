"""The package declares itself typed, and keeps declaring it.

``py.typed`` is an empty file whose only job is to exist. Nothing in this
repository breaks if it is deleted, every test still passes, and mypy still
runs clean here, because the annotations are all present either way. What
breaks is downstream: PEP 561 says a type checker must ignore a dependency's
inline annotations unless the package ships this marker, so without it every
consumer of plumbline sees ``Any`` and never finds out why.

That makes it the kind of file that gets lost in a refactor and is not missed
for a year. Hence a test.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import plumbline


def test_the_package_ships_a_py_typed_marker() -> None:
    package_root = Path(plumbline.__file__).parent

    assert (package_root / "py.typed").is_file(), (
        "py.typed is missing. Without it, PEP 561 requires type checkers to "
        "ignore this package's annotations, so downstream users silently get "
        "Any for everything plumbline exposes."
    )


def test_the_marker_is_empty_as_pep_561_intends() -> None:
    """The file is a flag, not a config. Content here would be a mistake."""
    marker = Path(plumbline.__file__).parent / "py.typed"

    assert marker.read_text(encoding="utf-8").strip() == ""


def test_the_version_is_a_release_version_not_a_placeholder() -> None:
    """Guards the mismatch that shipped in v0.1.0.

    The release was tagged v0.1.0 while the package still called itself
    0.1.0.dev0, and the test that should have caught it asserted a substring.
    """
    declared = tomllib.loads(
        (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    assert plumbline.__version__ == declared
    assert re.fullmatch(r"\d+\.\d+\.\d+", plumbline.__version__), plumbline.__version__
