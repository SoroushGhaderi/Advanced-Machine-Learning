#!/usr/bin/env python3
"""Structural and execution-state checks for the generated course."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
EXPECTED = [
    "00_course_setup_and_dataset.ipynb",
    "01_gradient_boosting_fundamentals.ipynb",
    "02_advanced_feature_engineering.ipynb",
    "03_imbalanced_learning.ipynb",
    "04_optuna_hyperparameter_optimization.ipynb",
    "05_ensemble_learning.ipynb",
    "06_anomaly_detection_extension.ipynb",
    "07_end_to_end_production_ml_project.ipynb",
]
REQUIRED_MARKERS = [
    "Learning objectives",
    "Estimated time",
    "Prerequisites",
    "Common mistakes and leakage warnings",
    "Exercises",
    "Challenge",
    "Summary",
    "References",
]


def main(names: list[str] | None = None) -> None:
    assert [p.name for p in NOTEBOOKS] == EXPECTED, "notebook sequence mismatch"
    selected = [ROOT / "notebooks" / name for name in names] if names else NOTEBOOKS
    unknown = [path.name for path in selected if path.name not in EXPECTED]
    assert not unknown, f"unknown notebooks: {unknown}"
    for path in selected:
        with path.open(encoding="utf-8") as handle:
            json.load(handle)  # explicit JSON validation
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        markdown = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")
        for marker in REQUIRED_MARKERS:
            assert marker.lower() in markdown.lower(), f"{path.name}: missing {marker}"
        code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
        assert code_cells, f"{path.name}: no code"
        errors = [output for cell in code_cells for output in cell.get("outputs", [])
                  if output.get("output_type") == "error"]
        assert not errors, f"{path.name}: contains stored execution errors"
        counts = [cell.execution_count for cell in code_cells]
        non_null_counts = [count for count in counts if count is not None]
        assert len(non_null_counts) in {0, len(counts)}, (
            f"{path.name}: partially executed notebook; clear or execute all code cells")
        assert len(non_null_counts) == len(set(non_null_counts)), (
            f"{path.name}: duplicate execution counts")
        if non_null_counts:
            assert non_null_counts == sorted(non_null_counts), (
                f"{path.name}: out-of-order execution counts")
    print(f"validated {len(selected)} notebooks; all checks passed")


if __name__ == "__main__":
    main(sys.argv[1:])
