from __future__ import annotations

import os
from pathlib import Path


def configure_runtime() -> Path:
    """Ensure runtime caches write to repo-local paths."""
    repo_root = Path(__file__).resolve().parents[1]
    os.environ["HOME"] = str(repo_root)

    mpl_dir = repo_root / ".cache" / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    (repo_root / "arviz_data").mkdir(parents=True, exist_ok=True)
    return repo_root
