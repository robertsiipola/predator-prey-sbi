from __future__ import annotations

from pathlib import Path


def load_lynx_hare(path: str) -> tuple[list[float], list[float], list[float]]:
    """Return (years, hare, lynx) arrays from a whitespace-delimited file."""
    file_path = Path(path)
    years: list[float] = []
    hare: list[float] = []
    lynx: list[float] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 3:
                raise ValueError(f"Expected 3 columns per line, got: {parts}")
            years.append(float(parts[0]))
            hare.append(float(parts[1]))
            lynx.append(float(parts[2]))

    if not years:
        raise ValueError(f"No data found in {file_path}")

    return years, hare, lynx
