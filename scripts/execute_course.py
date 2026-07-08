#!/usr/bin/env python3
"""Execute notebooks in isolated kernels and store their real outputs."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def execute(path: Path) -> None:
    started = time.perf_counter()
    with path.open(encoding="utf-8") as handle:
        nb = nbformat.read(handle, as_version=4)
    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="advanced-ml-course",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    client.execute()
    with path.open("w", encoding="utf-8") as handle:
        nbformat.write(nb, handle)
    print(f"executed {path.name} in {time.perf_counter() - started:.1f}s")


def main(names: list[str]) -> None:
    paths = [NB_DIR / name for name in names] if names else sorted(NB_DIR.glob("*.ipynb"))
    for path in paths:
        execute(path)


if __name__ == "__main__":
    main(sys.argv[1:])
