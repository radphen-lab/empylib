from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import sys


def repo_root(start: Path | None = None) -> Path:
    """Find the repository root by walking up until both `empylib/` and `docs/` exist."""
    current = (start or Path.cwd()).resolve()
    current = current if current.is_dir() else current.parent

    for candidate in (current, *current.parents):
        if (candidate / "empylib").exists() and (candidate / "docs").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the repository root. Expected a parent directory containing both 'empylib' and 'docs'."
    )


def ensure_repo_on_path(start: Path | None = None) -> Path:
    """Return the repository root and prepend it to sys.path if needed."""
    root = repo_root(start)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def docs_dir(start: Path | None = None) -> Path:
    return ensure_repo_on_path(start) / "docs"


def data_path(*parts: str, start: Path | None = None) -> Path:
    return docs_dir(start) / Path(*parts)


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(previous)
