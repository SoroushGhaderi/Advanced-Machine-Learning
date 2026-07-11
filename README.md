# Advanced Production-Oriented Machine Learning

A hands-on, notebook-first course for learners who already know Python,
pandas, scikit-learn, basic supervised learning, and introductory statistics.
The notebooks evolve one tabular classification project from problem framing
through final evaluation and production monitoring.

## Project and dataset decision

The course uses the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
(45,211 rows, mixed numeric/categorical predictors, imbalanced target). We chose
it over UCI Adult and UCI Online Shoppers because the campaign history supports
one coherent workflow from problem framing through feature engineering,
imbalance handling, explanation, and drift monitoring without exceeding laptop
scale.

The prediction moment is immediately before a scheduled call. `duration`, the
current call's duration, is unavailable at that point and is therefore excluded
from production models. Notebook 00 uses it once to demonstrate target leakage.
This is a prioritization project, not a causal estimate of whether calling
changes behavior. See [data/README.md](data/README.md) for the full data contract
and [notebooks/00_course_setup_and_dataset.ipynb](notebooks/00_course_setup_and_dataset.ipynb)
for the detailed comparison.

## Syllabus

| Notebook | Focus | Typical reduced-mode runtime |
|---|---|---:|
| `00_course_setup_and_dataset.ipynb` | framing, comparison, schema, leakage, split contract | < 20 s |
| `01_gradient_boosting_fundamentals.ipynb` | additive trees, gradients, shrinkage, early stopping | < 30 s |
| `02_advanced_feature_engineering.ipynb` | safe transformations, ablation, native categories | < 20 s |
| `03_imbalanced_learning.ipynb` | weights, thresholds, resampling, SMOTE/SMOTENC | < 45 s |
| `04_optuna_hyperparameter_optimization.ipynb` | Optuna objective design, search spaces, pruning, diagnostics | < 90 s |
| `05_ensemble_learning.ipynb` | Random Forest, voting, OOF stacking, complexity | < 60 s |
| `06_anomaly_detection_extension.ipynb` | optional controlled-contamination extension | < 20 s |
| `07_end_to_end_production_ml_project.ipynb` | final test, artifact, inference, model card | < 20 s |

Random Forest is taught as a bagging baseline inside ensembles. SMOTE is one
option inside imbalanced learning. Optuna is part of model development. Anomaly
detection is optional because the source data contain no natural anomaly label.

## Evaluation contract

- **Development (60%)**: fitting, cross-validation, features, and tuning.
- **Validation (20%)**: threshold selection and limited finalist comparison.
- **Test (20%)**: sealed until notebook 07 and used once for final evaluation.

The older Bank Marketing file lacks a complete timestamp and stable customer
ID. The course explains temporal and grouped validation but does not fabricate
either from insufficient fields. See [CONTEXT.md](CONTEXT.md) for canonical
terms and [data/README.md](data/README.md) for source and limitations.

## Setup

Python 3.11–3.13 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name advanced-ml-course --display-name "Advanced ML Course"
```

If standard `venv` is unavailable, install `virtualenv` and replace the first
command with `python3 -m virtualenv .venv`.

## Running the course

Laptop-friendly mode is the default:

```bash
export FAST_MODE=1
jupyter lab
```

Open and run notebooks in numeric order. `FAST_MODE=1` uses deterministic
stratified subsets, 3-fold CV, 8 Optuna trials, and smaller SHAP/ensemble
budgets. For the larger experiments:

```bash
export FAST_MODE=0
```

To reproduce the stored outputs from isolated kernels:

```bash
.venv/bin/python scripts/build_notebooks.py
FAST_MODE=1 .venv/bin/python scripts/execute_course.py
.venv/bin/python scripts/validate_course.py
```

The first notebook downloads the UCI archive and caches only the documented
CSV under `data/raw/`. No remote code is executed. Raw data, model artifacts,
and generated reports are Git-ignored.

## Learning path

1. Run 00 first; it defines the prediction contract, legal features, and data splits.
2. Run 01 to understand gradient boosting before advanced model development.
3. Run 02–04 to develop and evaluate candidates without test access.
4. Run 05–06 to decide whether complexity and the extension are worth it.
5. Run 07 once the workflow is frozen; it performs final test evaluation.

Every notebook includes objectives, time estimate, dependencies, explanations,
reproducible code, interpreted experiments, leakage warnings, exercises, a
challenge, summary, and authoritative references.

## Directory structure

```text
.
├── notebooks/              # executable course sequence
├── scripts/                # notebook builder, executor, validator
├── data/raw/               # cached source data (ignored)
├── models/                 # generated artifacts (ignored)
├── reports/                # generated metadata/reports (ignored)
├── data/README.md
├── CAPSTONE_RUBRIC.md
├── VALIDATION_REPORT.md
└── requirements.txt
```

## Capstone

Extend the final workflow with one justified feature or model change, evaluate
it without contaminating test evidence, document its explanations and
monitoring consequences, and submit the artifact plus report. Assessment uses
[CAPSTONE_RUBRIC.md](CAPSTONE_RUBRIC.md).

## Reproducibility limits

Seeds are fixed wherever supported. Parallel floating-point reductions and
library/platform differences can still create tiny numeric variation. The
stored notebook outputs were generated in reduced mode; full mode is expected
to take materially longer, especially Optuna and stacking.
