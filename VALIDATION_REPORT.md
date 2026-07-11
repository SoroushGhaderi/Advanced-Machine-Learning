# Validation Report

Validated on macOS arm64 with Python 3.13.1 in `FAST_MODE=1`.

## Ordered clean-kernel execution

All generated outputs were cleared, then every notebook was executed in
syllabus order with a new kernel per notebook.

| Notebook | Runtime | Result |
|---|---:|---|
| 00 | 6.3 s | Pass |
| 01 | 6.9 s | Pass |
| 02 | 6.8 s | Pass |
| 03 | 9.7 s | Pass |
| 04 | 16.1 s | Pass |
| 05 | 16.6 s | Pass |
| 06 | 5.2 s | Pass |
| 07 | 3.9 s | Pass |

The latest focused revalidation executed notebooks 05–07 in approximately 32
seconds, excluding environment installation. Stored outputs contain no error
outputs.

## Structural and boundary checks

`scripts/validate_course.py` confirmed:

- the focused notebooks 05–07 in both English and Farsi parse as JSON and pass
  `nbformat` validation;
- every code cell has an execution count and no stored error output;
- every notebook contains objectives, estimated time, prerequisites, leakage
  warnings, exercises, a challenge, summary, and references;
- notebooks 00–06 contain no final test scoring;
- notebook 03 uses imbalanced-learn pipelines and SMOTENC;
- notebook 04's Optuna objective contains no test access;
- notebook 05 contains explicit OOF prediction generation and scikit-learn's
  leakage-safe stacking implementation.

Manual/source audit additionally confirmed that learned preprocessing is inside
the relevant CV pipelines, each Optuna fold fits its own preprocessor, and the
test set is first scored in notebook 07.

## Reproducibility evidence

- Seed: 42.
- Source CSV SHA-256:
  `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d`.
- Reduced split sizes: 12,000 development / 4,000 validation / 4,000 test.
- Serialized pipeline reload matches pre-serialization probabilities to an
  absolute tolerance of `1e-12`.
- Final validation-selected threshold: 0.22.
- One-time reduced-mode test results: ROC-AUC 0.7803, PR-AUC 0.4222, log loss
  0.2940, precision 0.4468, recall 0.5021, specificity 0.9176.

These metrics describe the deterministic reduced-mode sample, not a claim about
future deployment performance.

## Failures and warnings encountered

No failures remain in the delivered execution. During implementation:

1. UCI's current download is a nested archive (`bank-marketing.zip` containing
   `bank.zip`); the loader was corrected to validate and extract only the
   documented nested `bank-full.csv` member.
2. Dummy-classifier undefined precision was made explicit with
   `zero_division=0`.
3. A logistic ensemble member was given numeric scaling to resolve convergence
   warnings.
4. Exact known LightGBM feature-name and optional Jupyter progress-widget
   interoperability warnings were suppressed after verifying they did not
   affect predictions. No broad warning suppression is used.

## Runtime-heavy sections

Reduced mode is intentionally modest. The heaviest notebooks are:

- 04: 8 Optuna trials × 3 folds plus LightGBM early stopping and study diagnostics;
- 05: OOF predictions, voting, and stacking with 180-tree Random Forest;
- 06: Isolation Forest and novelty-mode LOF on controlled contamination.

`FAST_MODE=0` raises CV to five folds, Optuna to 25 trials, ensemble tree counts,
and SHAP sample sizes. Full mode was not executed in this validation pass and is
expected to take materially longer.

## Assumptions and limitations

- Prediction occurs immediately before a scheduled call; `duration` is illegal.
- Illustrative business cost is `FP + 5 × FN`; the source does not provide real
  bank economics.
- No stable customer ID or full timestamp exists, so grouped and true temporal
  validation are discussed but not fabricated.
- The anomaly extension measures recovery of controlled perturbations, not
  real fraud or data-error detection.
- Seeds improve repeatability, but parallel floating-point/library differences
  can produce small numeric variation.
