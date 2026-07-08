#!/usr/bin/env python3
"""Build the course notebooks from readable cell definitions.

The generated notebooks contain no fabricated outputs. Use execute_course.py to
run them in order and store real outputs.
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def md(text: str) -> dict:
    """Create a markdown notebook cell from plain text."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": dedent(text).strip() + "\n"}


def code(text: str) -> dict:
    """Create a code notebook cell and strip external helper imports."""
    source = dedent(text).strip() + "\n"
    # Every notebook embeds the shared teaching helpers below.  Strip legacy
    # imports so a generated notebook can be copied out of this repository and
    # still run without the external ``src.course_utils`` module.
    source = re.sub(
        r"^from src\.course_utils import (?:\([^)]*\)|[^\n]+)\n?",
        "",
        source,
        flags=re.MULTILINE,
    )
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def notebook(cells: list[dict]) -> dict:
    """Wrap a list of cells in a notebook structure."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Advanced ML Course", "language": "python", "name": "advanced-ml-course"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON = code(
    """
    import hashlib, json, os, platform, random, warnings
    from pathlib import Path
    ROOT = Path.cwd()
    if not (ROOT / "data").exists() and (ROOT.parent / "data").exists():
        ROOT = ROOT.parent
    # Known interoperability/UI warnings do not affect predictions or notebook execution.
    warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier")
    warnings.filterwarnings("ignore", message="IProgress not found.*")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (balanced_accuracy_score, brier_score_loss, confusion_matrix,
                                 f1_score, log_loss, precision_score, recall_score)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SEED = 42
    TARGET = "y"
    LEAKAGE_COLUMNS = ["duration"]

    def project_root():
        '''Return the course root when present, otherwise the notebook directory.'''
        # Return the course root when present, otherwise the notebook's directory.
        return ROOT

    def set_seed(seed=SEED):
        '''Seed Python and NumPy RNGs for reproducible notebook runs.'''
        random.seed(seed)
        np.random.seed(seed)

    def fast_mode():
        '''Report whether notebooks should use the reduced fast-mode sample.'''
        # Set FAST_MODE=0 for full-size experiments; laptop mode is the default.
        return os.getenv("FAST_MODE", "1").lower() not in {"0", "false", "no"}

    def bank_data_path():
        '''Locate the bundled Bank Marketing CSV file.'''
        # The course ships with a local dataset; notebooks never access the network.
        path = project_root() / "data" / "raw" / "bank-full.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Expected the bundled Bank Marketing data at {path}. "
                "Run the notebook from the course root or place bank-full.csv there."
            )
        return path

    def file_sha256(path):
        '''Compute the SHA-256 digest of a local file.'''
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load_bank_data(include_duration=False):
        '''Load the Bank Marketing dataset and drop leakage columns by default.'''
        # Load the data, encode y, and exclude post-call duration by default.
        frame = pd.read_csv(bank_data_path(), sep=";")
        frame[TARGET] = frame[TARGET].map({"no": 0, "yes": 1}).astype("int8")
        if not include_duration:
            frame = frame.drop(columns=LEAKAGE_COLUMNS)
        return frame

    def stratified_sample(frame, n, seed=SEED):
        '''Draw a label-preserving sample from a classified dataset.'''
        if n >= len(frame):
            return frame.copy()
        fractions = frame[TARGET].value_counts(normalize=True)
        counts = (fractions * n).round().astype(int)
        counts.iloc[0] += n - counts.sum()
        parts = [group.sample(n=min(counts.loc[label], len(group)),
                              random_state=seed + int(label))
                 for label, group in frame.groupby(TARGET)]
        return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    def make_splits(frame=None, reduced=None):
        '''Create deterministic stratified train, validation, and test splits.'''
        # Deterministic stratified 60/20/20 split; test stays sealed until notebook 09.
        from sklearn.model_selection import train_test_split
        frame = load_bank_data() if frame is None else frame
        train_val, test = train_test_split(
            frame, test_size=0.20, stratify=frame[TARGET], random_state=SEED)
        train, validation = train_test_split(
            train_val, test_size=0.25, stratify=train_val[TARGET], random_state=SEED)
        reduced = fast_mode() if reduced is None else reduced
        if reduced:
            train = stratified_sample(train, 12_000)
            validation = stratified_sample(validation, 4_000, SEED + 1)
            test = stratified_sample(test, 4_000, SEED + 2)
        return tuple(part.reset_index(drop=True) for part in (train, validation, test))

    def split_xy(frame):
        '''Separate a frame into feature matrix and target vector.'''
        return frame.drop(columns=TARGET), frame[TARGET]

    def feature_groups(frame):
        '''Identify numeric and categorical feature columns.'''
        features = frame.drop(columns=[TARGET], errors="ignore")
        categorical = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        numerical = features.select_dtypes(include=np.number).columns.tolist()
        return numerical, categorical

    def make_preprocessor(frame, scale_numeric=True):
        '''Build the preprocessing pipeline for numeric and categorical features.'''
        # Preprocessing is fitted only inside the enclosing training/CV pipeline.
        numerical, categorical = feature_groups(frame)
        numeric_steps = [("impute", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scale", StandardScaler()))
        categorical_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     min_frequency=10, sparse_output=True)),
        ])
        return ColumnTransformer([
            ("numeric", Pipeline(numeric_steps), numerical),
            ("categorical", categorical_pipe, categorical),
        ], sparse_threshold=0.3)

    def classification_metrics(y_true, probability, threshold=0.5):
        '''Compute ranking and threshold-based classification metrics.'''
        prediction = np.asarray(probability) >= threshold
        tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
        return {"log_loss": log_loss(y_true, probability),
                "brier_score": brier_score_loss(y_true, probability),
                "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
                "f1": f1_score(y_true, prediction, zero_division=0),
                "precision": precision_score(y_true, prediction, zero_division=0),
                "recall": recall_score(y_true, prediction, zero_division=0),
                "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
                "cost": float(fp + 5 * fn)}

    def threshold_table(y_true, probability, thresholds=None):
        '''Evaluate classification metrics across a list of decision thresholds.'''
        thresholds = np.linspace(0.05, 0.80, 76) if thresholds is None else thresholds
        return pd.DataFrame([{"threshold": float(t),
                              **classification_metrics(y_true, probability, float(t))}
                             for t in thresholds])

    def add_domain_features(frame):
        '''Create domain-inspired features for the Bank Marketing dataset.'''
        result = frame.copy()
        result["was_previously_contacted"] = (result["pdays"] != -1).astype("int8")
        result["pdays_clean"] = result["pdays"].replace(-1, np.nan)
        result["contact_pressure"] = result["campaign"] / (1 + result["previous"])
        result["balance_per_age"] = result["balance"] / result["age"].clip(lower=18)
        result["age_band"] = pd.cut(result["age"], bins=[0, 29, 39, 49, 59, np.inf],
                                    labels=["<30", "30s", "40s", "50s", "60+"]).astype("object")
        return result.drop(columns=["pdays"])

    def environment_metadata():
        '''Collect runtime metadata for experiment tracking.'''
        import sklearn
        return {"python": platform.python_version(), "platform": platform.platform(),
                "numpy": np.__version__, "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__}

    def write_json(data, path):
        '''Serialize structured data to a JSON file on disk.'''
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    set_seed(SEED)
    FAST_MODE = fast_mode()
    CV_FOLDS = 3 if FAST_MODE else 5
    sns.set_theme(style="whitegrid", context="notebook")
    pd.set_option("display.max_columns", 30)
    print({"FAST_MODE": FAST_MODE, "CV_FOLDS": CV_FOLDS, "seed": SEED})
    """
)

COMMON_NO_LGBM_WARNING = {
    **COMMON,
    "id": uuid.uuid4().hex[:8],
    "source": COMMON["source"].replace(
        'warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier")\n',
        "",
    ),
}


def n00() -> list[dict]:
    """Build notebook 00: course setup and dataset exploration."""
    return [
        md(
            """
            # 00 — Course setup and dataset

            **Estimated time:** 60–90 minutes
            **Prerequisites:** Python, pandas, scikit-learn basics, and introductory statistics.
            **Depends on:** nothing.

            ## Learning objectives

            - Frame the prediction problem at a precise decision moment.
            - Load the course dataset from local files, not the network.
            - Distinguish between a data catalog and a data dictionary.
            - Spot leakage before modeling starts.
            - Build a simple leakage-safe baseline with scikit-learn.
            - Define development, validation, and sealed test splits.

            The question for the course is: **before a call is placed, which clients should be
            prioritized for a term-deposit marketing campaign?** This is a ranking-and-selection
            problem, not a causal-effect problem.

            **Teaching note:** students often jump straight to model tuning. Here we slow down and
            treat dataset choice, prediction moment, and split design as first-class modeling steps.
            """
        ),
        COMMON,
        md(
            """
            ## Why this dataset

            We want a dataset that supports practical teaching:

            - a realistic binary target,
            - mixed numeric and categorical fields,
            - clear leakage risks,
            - enough size to show validation patterns,
            - and a business context students can explain.

            | Dataset | Why it works | Main drawback |
            |---|---|---|
            | UCI Bank Marketing | realistic campaign data, class imbalance, and obvious leakage risk | `duration` is post-call information |
            | UCI Adult | useful for preprocessing practice | less aligned with a campaign decision workflow |
            | UCI Online Shoppers | easy to understand conversion task | less instructive for campaign history and governance |

            **Recommendation:** use Bank Marketing. It gives us a clean path through cataloging,
            leakage checks, preprocessing, baseline modeling, and operational judgment.
            """
        ),
        code(
            """
            path = bank_data_path()
            raw = load_bank_data(include_duration=True)
            print("cached path:", path)
            print("shape:", raw.shape)
            raw.head()
            """
        ),
        md(
            """
            ## Data catalog: making the dataset findable and governable

            A **data catalog** records the context needed to discover, understand, trust, and govern
            a data asset. A **data dictionary** is narrower: it explains individual fields.

            In class, I like to treat the catalog as the answer to three questions:

            1. What is this data for?
            2. When is it safe to use?
            3. What should stop us from using it?

            ![A raw dataset becomes a governed data asset through catalog metadata, enabling discovery, leakage-safe machine learning, and governance.](../assets/data_catalog_learning_flow.png)

            *Teaching note:* the catalog is not just documentation. It is part of the modeling contract.
            """
        ),
        code(
            """
            asset_catalog = pd.Series({
                "asset_name": "uci_bank_marketing",
                "local_path": str(path.relative_to(ROOT)),
                "source_url": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
                "sha256": file_sha256(path),
                "rows": len(raw),
                "columns": raw.shape[1],
                "grain": "one row per marketing contact",
                "target": TARGET,
                "prediction_moment": "immediately before a scheduled call",
                "refresh_policy": "static course snapshot",
                "owner": "course maintainer",
                "steward": "simulated marketing-data steward",
            }, name="value").to_frame()
            display(asset_catalog)
            """
        ),
        md(
            """
            ## Field catalog and data dictionary

            UCI documents `unknown` as an actual category; it is not silently converted to missing.
            `pdays == -1` is a sentinel meaning the client was not previously contacted. `y=1`
            means the client subscribed. Several fields describe the current campaign; availability
            must be checked against the prediction moment rather than inferred from the column name.

            The field catalog adds two useful governance properties:

            - the field's role in the process,
            - and whether the field is available at prediction time.

            That makes the table behave like a feature contract, not just a schema dump.
            """
        ),
        code(
            """
            field_roles = {
                "age": "client attribute", "job": "client attribute", "marital": "client attribute",
                "education": "client attribute", "default": "financial attribute",
                "balance": "financial attribute", "housing": "financial attribute", "loan": "financial attribute",
                "contact": "campaign context", "day": "campaign context", "month": "campaign context",
                "duration": "post-contact measurement", "campaign": "campaign history",
                "pdays": "campaign history", "previous": "campaign history", "poutcome": "campaign history",
                TARGET: "target",
            }
            schema = pd.DataFrame({
                "role": [field_roles[c] for c in raw.columns],
                "dtype": raw.dtypes.astype(str),
                "missing": raw.isna().sum(),
                "unique": raw.nunique(),
                "available_at_prediction": [c != "duration" and c != TARGET for c in raw.columns],
                "example": [raw[c].iloc[0] for c in raw.columns],
            })
            display(schema)
            print("target distribution:")
            display(raw[TARGET].value_counts().rename("count").to_frame().assign(rate=lambda x: x["count"] / len(raw)))
            """
        ),
        code(
            """
            categorical = raw.select_dtypes(include="object").columns.tolist()
            numeric = raw.select_dtypes(include=np.number).columns.drop(TARGET).tolist()
            print("categorical:", categorical)
            print("numeric:", numeric)
            display(raw[categorical].nunique().sort_values().rename("cardinality").to_frame())
            """
        ),
        md(
            """
            ### Minimum quality contract

            Catalog descriptions are promises, not evidence. A useful catalog turns each important
            promise into an explicit check, so we can tell the difference between a trustworthy
            snapshot and a dataset that only looks clean at first glance.

            For this snapshot, the minimum contract is intentionally small and strict:

            - `y` must be binary and non-null, because the target is the label we train and evaluate
              against.
            - `age` must stay in a plausible adult range, so obvious data-entry errors surface early.
            - `day` and `month` must form valid calendar fields, because downstream feature logic
              assumes a real call date.
            - `pdays` must obey the documented sentinel rule: `-1` means the client was not
              previously contacted, while non-negative values represent elapsed days since the
              last campaign contact.

            These checks are not cosmetic. They are the boundary between a dataset we can trust and
            one we should quarantine before it reaches modeling or reporting.

            **Warning:** if a critical rule breaks in production data, the correct response is to fail
            ingestion or quarantine the batch. Do not silently weaken the rule to make a pipeline pass.
            """
        ),
        code(
            """
            quality_checks = pd.Series({
                "target is binary and non-null": raw[TARGET].notna().all() and set(raw[TARGET].unique()) <= {0, 1},
                "age is between 18 and 100": raw["age"].between(18, 100).all(),
                "day is between 1 and 31": raw["day"].between(1, 31).all(),
                "month uses documented abbreviations": set(raw["month"]) <= set("jan feb mar apr may jun jul aug sep oct nov dec".split()),
                "pdays is -1 or non-negative": ((raw["pdays"] == -1) | (raw["pdays"] >= 0)).all(),
                "source row count is stable": len(raw) == 45_211,
            }, name="passed").to_frame()
            display(quality_checks)
            assert quality_checks["passed"].all(), "Critical catalog quality contract failed"
            """
        ),
        md(
            """
            ## Leakage audit: why `duration` must go

            `duration` is only known during or after the call, so a pre-contact system cannot use it.
            This is a classic leakage trap: the feature looks highly predictive, but only because it
            contains information from the event we are trying to predict.

            The next cell compares a simple logistic-regression pipeline with and without `duration`.
            If the score jumps a lot, that is a warning sign, not a deployment win.
            """
        ),
        code(
            """
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold, cross_validate
            from sklearn.pipeline import Pipeline

            audit = stratified_sample(raw, 12_000 if FAST_MODE else len(raw))
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

            def leakage_audit(frame):
                X, y = frame.drop(columns=TARGET), frame[TARGET]
                model = Pipeline([
                    ("preprocess", make_preprocessor(frame)),
                    ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
                ])
                result = cross_validate(
                    model,
                    X,
                    y,
                    cv=cv,
                    scoring=["neg_log_loss", "balanced_accuracy"],
                    n_jobs=-1,
                )
                return {
                    "log_loss (lower is better)": -result["test_neg_log_loss"].mean(),
                    "balanced_accuracy (higher is better)": result["test_balanced_accuracy"].mean(),
                }

            leakage_comparison = pd.DataFrame({
                "with_duration (invalid)": leakage_audit(audit),
                "without_duration (deployable)": leakage_audit(audit.drop(columns="duration")),
            }).T
            leakage_comparison
            """
        ),
        md(
            """
            **Interpretation:** `duration` usually creates a dramatic apparent gain because successful
            calls tend to last longer.

            Other risks are subtler:

            - `campaign` includes the current contact,
            - `day` and `month` identify the current contact date,
            - and `pdays`, `previous`, `poutcome` describe prior contact history.

            We keep those fields only under the explicit assumption that scoring happens immediately
            before a scheduled call. Change the decision moment and the feature contract must change too.
            """
        ),
        md(
            """
            ## The split contract

            - **Development (60%)**: fitting, cross-validation, feature/model/hyperparameter selection.
            - **Validation (20%)**: operating-threshold selection and limited finalist comparison.
            - **Test (20%)**: sealed until notebook 09; used once for final evaluation.

            A true temporal split would be preferable for deployment forecasting, but this file lacks a
            year and stable full timestamp. A grouped split would help with repeated clients, but no
            customer identifier is supplied.

            **Best-practice warning:** do not use validation or test data to choose features, thresholds,
            preprocessing choices, or the model family. That is how notebooks accidentally become less
            trustworthy than they look.
            """
        ),
        code(
            """
            clean = load_bank_data()  # duration excluded by default
            development, validation, test = make_splits(clean, reduced=FAST_MODE)
            split_summary = pd.DataFrame({
                "rows": [len(development), len(validation), len(test)],
                "positive_rate": [x[TARGET].mean() for x in (development, validation, test)],
            }, index=["development", "validation", "test (sealed)"])
            display(split_summary)
            assert "duration" not in clean.columns
            assert sum(len(x) for x in make_splits(clean, reduced=False)) == len(clean)
            """
        ),
        md(
            """
            ## Practical baseline

            A strong course notebook should not stop at data inspection. We also want a small
            deployable baseline that students can understand, modify, and compare against later
            models.

            The baseline below uses:

            - the leakage-safe columns only,
            - a standard preprocessing pipeline,
            - and logistic regression as a conventional first model.

            This is intentionally simple. If a sophisticated method cannot beat this setup, we should
            be cautious about adding complexity.
            """
        ),
        code(
            """
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, log_loss
            from sklearn.model_selection import StratifiedKFold, cross_val_predict
            from sklearn.pipeline import Pipeline

            def split_xy(frame):
                return frame.drop(columns=TARGET), frame[TARGET]

            baseline = Pipeline([
                ("preprocess", make_preprocessor(development)),
                ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
            ])

            X_dev, y_dev = split_xy(development)
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
            dev_prob = cross_val_predict(baseline, X_dev, y_dev, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

            baseline_summary = pd.Series({
                "log_loss": log_loss(y_dev, dev_prob),
                "brier_score": brier_score_loss(y_dev, dev_prob),
                "balanced_accuracy@0.5": balanced_accuracy_score(y_dev, dev_prob >= 0.5),
                "f1@0.5": f1_score(y_dev, dev_prob >= 0.5, zero_division=0),
            }, name="development CV")
            display(baseline_summary.to_frame())
            """
        ),
        md(
            """
            ## Evaluation and interpretation

            For imbalanced problems, accuracy alone is usually misleading. We prefer:

            - log loss and Brier score for probability quality,
            - precision and recall for positive-class usefulness,
            - balanced accuracy for a simple class-balance check,
            - and cost for the operational trade-off we actually care about.

            **Teaching note:** if students ask why we do not use the test set here, that is a good sign.
            The answer is that we are still choosing the workflow, so we stay on development data.
            """
        ),
        code(
            """
            dev_pred = (dev_prob >= 0.5).astype(int)
            interpretation = pd.DataFrame({
                "predicted_positive_rate": [dev_pred.mean()],
                "actual_positive_rate": [y_dev.mean()],
                "precision": [precision_score(y_dev, dev_pred, zero_division=0)],
                "recall": [recall_score(y_dev, dev_pred, zero_division=0)],
            }, index=["development CV"])
            display(interpretation)
            """
        ),
        md(
            """
            ## Practical recommendation

            Use Bank Marketing as the course dataset, but keep the prediction contract strict:

            - exclude `duration`,
            - document when each field is available,
            - use pipelines so preprocessing stays inside cross-validation,
            - and keep the test set sealed until the final evaluation notebook.

            If we later improve the model, the main gain should come from better feature design,
            threshold selection, and validation discipline, not from hiding leakage.
            """
        ),
        md(
            """
            ## Exercises

            1. Rewrite the asset catalog for a production setting with a named owner and freshness SLA.
            2. Add one cross-field rule involving `pdays`, `previous`, and `poutcome`.
            3. Change the prediction moment to "one week before the campaign" and identify which fields
               become unavailable.
            4. Replace logistic regression with a dummy classifier and explain why it is still a useful baseline.
            5. Propose one evaluation metric you would show a marketing stakeholder and one you would show an ML engineer.

            ## Summary

            We selected a dataset, documented its contract, checked basic quality rules, audited leakage,
            and built a small leakage-safe baseline. That sets up the rest of the course with a common
            language for features, splits, and evaluation.

            ## References

            - [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
            - [Dataset paper (Decision Support Systems)](https://doi.org/10.1016/j.dss.2014.03.001)
            - [Data Catalog Vocabulary (DCAT) v3](https://www.w3.org/TR/vocab-dcat-3/)
            - [scikit-learn model selection](https://scikit-learn.org/stable/model_selection.html)
            """
        ),
    ]


def n01() -> list[dict]:
    """Build notebook 01: evaluation and cross-validation."""
    return [
        md(
            """
            # 01 — Evaluation and cross-validation

            **Estimated time:** 90–120 minutes  
            **Prerequisites:** notebook 00 and classification metrics.  
            **Depends on:** the prediction moment and split contract from notebook 00.

            ## Learning objectives

            - Build dummy and logistic baselines without touching the test set.
            - Use stratified cross-validation for model selection.
            - connect discrimination, probability quality, classification metrics, and business cost.
            - Select a threshold on validation data and assess calibration.
            """
        ),
        COMMON,
        code(
            """
            from sklearn.model_selection import StratifiedKFold, cross_validate
            from sklearn.pipeline import Pipeline
            from sklearn.dummy import DummyClassifier
            from sklearn.linear_model import LogisticRegression
            from src.course_utils import load_bank_data, make_splits, make_preprocessor, split_xy

            development, validation, _sealed_test = make_splits(load_bank_data(), reduced=FAST_MODE)
            X_dev, y_dev = split_xy(development)
            X_val, y_val = split_xy(validation)
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
            print(development.shape, validation.shape, "test remains unused")
            """
        ),
        md(
            """
            ## Baselines before sophistication

            A prevalence dummy anchors probability metrics; a most-frequent dummy anchors hard-label
            accuracy. Logistic regression supplies a transparent deployable baseline. Every learned
            preprocessing step is inside its pipeline and therefore refit within each CV training fold.
            """
        ),
        code(
            """
            preprocess = make_preprocessor(development)
            models = {
                "dummy_prior": Pipeline([("preprocess", preprocess), ("model", DummyClassifier(strategy="prior"))]),
                "logistic": Pipeline([("preprocess", make_preprocessor(development)),
                                      ("model", LogisticRegression(max_iter=1200, random_state=SEED))]),
            }
            from sklearn.metrics import f1_score, make_scorer, precision_score
            scoring = {"accuracy": "accuracy", "precision": make_scorer(precision_score, zero_division=0), "recall": "recall",
                       "f1": make_scorer(f1_score, zero_division=0), "balanced_accuracy": "balanced_accuracy",
                       "neg_log_loss": "neg_log_loss", "neg_brier_score": "neg_brier_score"}
            rows = []
            for name, model in models.items():
                scores = cross_validate(model, X_dev, y_dev, cv=cv, scoring=scoring, n_jobs=-1)
                rows.append({"model": name, **{k: (-scores[f"test_{k}"].mean() if k.startswith("neg_") else scores[f"test_{k}"].mean())
                                               for k in scoring}})
            cv_results = pd.DataFrame(rows).set_index("model").rename(columns={
                "neg_log_loss": "log_loss",
                "neg_brier_score": "brier_score",
            })
            cv_results
            """
        ),
        md(
            """
            Accuracy can exceed 88% by predicting no subscription for everyone. Balanced accuracy gives
            equal weight to the positive and negative classes, so the majority class cannot dominate the
            summary. F1 combines precision and recall for the positive class, while log loss and Brier score
            evaluate the quality of predicted probabilities.

            For threshold $t$, recall is $TP/(TP+FN)$, specificity is $TN/(TN+FP)$, and precision is
            $TP/(TP+FP)$. No threshold is universally correct: it encodes an operating trade-off.
            """
        ),
        code(
            """
            logistic = models["logistic"].fit(X_dev, y_dev)
            val_probability = logistic.predict_proba(X_val)[:, 1]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            threshold_preview = threshold_table(y_val, val_probability)
            threshold_preview.plot(x="threshold", y=["precision", "recall", "f1"], ax=axes[0])
            axes[0].set_title("Validation threshold metrics")
            threshold_preview.plot(x="threshold", y="cost", ax=axes[1], legend=False)
            axes[1].set_title("Validation cost by threshold")
            axes[1].set_ylabel("cost")
            plt.tight_layout()
            """
        ),
        md(
            """
            ## Threshold selection from an explicit cost

            We use a teaching assumption: a wasted call (false positive) costs 1 unit and a missed
            subscriber (false negative) costs 5. Absolute values are fictional; the discipline of stating
            them is the lesson. Validation chooses the threshold. The test set remains sealed.
            """
        ),
        code(
            """
            from src.course_utils import classification_metrics, threshold_table
            table = threshold_table(y_val, val_probability)
            best = table.loc[table["cost"].idxmin()]
            display(table.sort_values("cost").head(8))
            print("selected threshold:", round(best["threshold"], 3))
            display(pd.DataFrame({
                "threshold_0.5": classification_metrics(y_val, val_probability, 0.5),
                "cost_selected": classification_metrics(y_val, val_probability, best["threshold"]),
            }).T)
            """
        ),
        code(
            """
            from sklearn.metrics import ConfusionMatrixDisplay
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            for ax, (label, threshold) in zip(axes, [("0.50", .5), ("cost-selected", best["threshold"])]):
                ConfusionMatrixDisplay.from_predictions(y_val, val_probability >= threshold, ax=ax, colorbar=False)
                ax.set_title(label)
            plt.tight_layout()
            """
        ),
        md("## Calibration: do predicted probabilities mean what they say?"),
        code(
            """
            from sklearn.calibration import CalibrationDisplay
            from sklearn.metrics import brier_score_loss
            CalibrationDisplay.from_predictions(y_val, val_probability, n_bins=10, strategy="quantile")
            plt.title("Validation calibration (quantile bins)")
            plt.show()
            print("Brier score:", round(brier_score_loss(y_val, val_probability), 4))
            """
        ),
        md(
            """
            Calibration and hard-label performance answer different questions. A model can choose useful
            clients at one threshold while systematically overstating probabilities. A calibration curve is
            itself an estimate: small bins are noisy, and calibration fitted on the same data is optimistic.

            ## Common mistakes and leakage warnings

            - Choosing a metric because it gives the largest-looking number.
            - Selecting the threshold independently in every CV fold and then averaging incomparable labels.
            - Calibrating and evaluating calibration on the same observations.
            - Repeatedly checking test performance while refining a model.
            - Assuming stratification solves temporal or customer-group leakage.

            ## Exercises

            1. Recompute the threshold for FN costs of 2, 10, and 20; plot the selected threshold.
            2. Compare fixed-width and quantile calibration bins.
            3. **Challenge:** define a capacity-constrained policy that can call only 10% of clients.
               Evaluate precision@10%, recall@10%, and expected cost.

            ## Summary

            Cross-validation selected the model family; validation selected an operating threshold.
            F1, balanced accuracy, and business cost exposed behavior hidden by accuracy, while calibration assessed the
            meaning of probabilities. The test set has still not been scored.

            ## References

            - [scikit-learn classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
            - [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
            - [scikit-learn Brier score loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html)
            """
        ),
    ]


def n01_gradient_boosting() -> list[dict]:
    """Assemble the gradient boosting fundamentals notebook."""
    return [
        md(
            """
            # 01 — Gradient boosting fundamentals

            **Estimated time:** 100–130 minutes  
            **Prerequisites:** notebook 00; decision trees and basic classification metrics.  
            **Depends on:** the pre-contact feature contract and development/validation split.

            ## Learning objectives

            - Explain boosting as a sequential additive model rather than a collection of independent trees.
            - Show why each new tree follows the negative gradient of a loss function.
            - Connect residual fitting in regression to probability errors in binary classification.
            - Interpret `n_estimators`, `learning_rate`, and `max_depth` as a capacity tradeoff.
            - Evaluate gradient boosting against simple baselines without touching the sealed test set.

            ## The central idea

            A decision tree is a weak learner when deliberately kept shallow. Gradient boosting combines many
            such trees **sequentially**. Tree 1 makes a rough prediction; tree 2 learns where the current
            ensemble is wrong; tree 3 corrects the remaining error; and so on. Unlike a random forest, these
            trees are dependent—later trees cannot be trained before earlier trees exist.

            The word *gradient* comes from optimization. At each round, the algorithm calculates the direction
            that most rapidly reduces the chosen loss and fits a small tree to approximate that direction.
            The learning rate shrinks every correction, trading faster learning for stronger regularization.

            ## The training process, step by step

            1. **Initialize the ensemble.** For binary classification, every row begins with the same constant
               score, usually derived from the positive-class rate. The logistic function converts that score
               into an initial probability.
            2. **Measure the current errors.** Compare each observed label with its current probability. Under
               binary log loss, `y - p` gives the direction of the required correction: positive values push
               the score upward and negative values push it downward.
            3. **Fit a shallow correction tree.** The new tree partitions rows with similar error signals. Its
               leaves estimate how the current ensemble should change within each region.
            4. **Shrink the correction.** Multiply the tree output by `learning_rate`. Small updates make
               learning slower but reduce the chance that one tree overreacts to noise.
            5. **Update the ensemble.** Add the shrunken correction to the existing score. The new probability
               comes from the sum of the initial score and every tree contribution learned so far.
            6. **Repeat.** Recalculate errors using the updated ensemble, fit the next tree, and continue until
               the tree budget is exhausted or holdout loss stops improving.

            During prediction, the model does not calculate training errors again. A row passes through every
            learned tree; their shrunken outputs are summed with the initial score, converted to a probability,
            and compared with the decision threshold.
            """
        ),
        md(
            """
            ### Gradient boosting workflow at a glance

            ![Gradient boosting workflow: begin with a constant prediction, calculate probability errors, fit a shallow correction tree, shrink its output, update the ensemble, and repeat sequentially.](../assets/gradient_boosting_process.png)

            Read the upper process from left to right, then follow the loop back from step 6: after every
            update, the next tree sees newly calculated errors. The lower comparison highlights the structural
            difference from a random forest. Random-forest trees are independent and can train in parallel;
            boosting trees form an ordered chain because each tree depends on the current ensemble. The values
            shown are pedagogical examples rather than outputs from the Bank Marketing experiment.
            """
        ),
        COMMON,
        code(
            """
            from sklearn.compose import ColumnTransformer
            from sklearn.dummy import DummyClassifier
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                         log_loss, ConfusionMatrixDisplay)
            from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            from sklearn.tree import DecisionTreeRegressor
            from src.course_utils import load_bank_data, make_splits, split_xy

            development, validation, _sealed_test = make_splits(load_bank_data(), reduced=FAST_MODE)
            X_dev, y_dev = split_xy(development)
            X_val, y_val = split_xy(validation)
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
            CV_SPLITS = list(cv.split(X_dev, y_dev))
            print({"development": X_dev.shape, "validation": X_val.shape,
                   "positive_rate": round(y_dev.mean(), 3)})
            """
        ),
        md(
            """
            ## Build boosting by hand: squared-error regression

            Regression gives the cleanest intuition. Under squared-error loss, the negative gradient is the
            familiar residual `actual - prediction`. We start with the mean target, fit a shallow tree to the
            residuals, shrink its output, add it to the current prediction, and repeat.

            \\[
            F_m(x) = F_{m-1}(x) + \\eta h_m(x)
            \\]

            Here, `h_m` is the new tree and `η` is the learning rate. The ensemble—not the newest tree—is the
            model.
            """
        ),
        code(
            """
            rng = np.random.default_rng(SEED)
            x = np.linspace(-3, 3, 160)
            y = np.sin(x) + 0.25 * x + rng.normal(0, 0.12, len(x))
            X_toy = x.reshape(-1, 1)

            learning_rate = 0.25
            prediction = np.full_like(y, y.mean())
            snapshots = {0: prediction.copy()}
            toy_trees = []

            for round_number in range(1, 11):
                residual = y - prediction
                tree = DecisionTreeRegressor(max_depth=1, random_state=SEED + round_number)
                tree.fit(X_toy, residual)
                prediction += learning_rate * tree.predict(X_toy)
                toy_trees.append(tree)
                if round_number in {1, 3, 10}:
                    snapshots[round_number] = prediction.copy()

            fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
            for axis, (round_number, stage_prediction) in zip(axes, snapshots.items()):
                axis.scatter(x, y, s=9, alpha=0.35, label="observations")
                axis.plot(x, stage_prediction, color="tab:orange", linewidth=2, label="ensemble")
                axis.set_title(f"after {round_number} trees")
                axis.set_xlabel("x")
            axes[0].set_ylabel("target")
            axes[0].legend(loc="upper left")
            fig.suptitle("Each stump adds a small correction to the ensemble", y=1.04)
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            We can also look at the *errors* directly. Boosting does not replace the whole model at each
            step; it keeps the current ensemble and asks the next stump to explain what is still left in the
            residuals.
            """
        ),
        code(
            """
            # Compare the first correction step with the final residuals.
            residual_0 = y - snapshots[0]
            first_tree = toy_trees[0]
            first_correction = learning_rate * first_tree.predict(X_toy)
            residual_1 = y - snapshots[1]
            residual_final = y - snapshots[10]

            fig, axes = plt.subplots(2, 2, figsize=(14, 6.2), sharex=True)
            axes = axes.ravel()

            axes[0].scatter(x, residual_0, s=9, alpha=0.35, color="tab:blue")
            axes[0].axhline(0, color="black", linewidth=1)
            axes[0].set_title("start: residuals from mean")

            axes[1].scatter(x, residual_0, s=9, alpha=0.18, color="tab:blue", label="residual before 1 tree")
            axes[1].scatter(x, residual_1, s=9, alpha=0.35, color="tab:green", label="residual after 1 tree")
            axes[1].plot(x, first_correction, color="tab:orange", linewidth=2, label="first tree correction")
            axes[1].axhline(0, color="black", linewidth=1)
            axes[1].set_title("one stump reduces the error")

            step_idx = np.linspace(10, len(x) - 10, 10, dtype=int)
            axes[2].plot(x, snapshots[0], color="tab:gray", linewidth=1.4, label="initial prediction")
            axes[2].plot(x, snapshots[1], color="tab:orange", linewidth=2, label="after 1 tree")
            axes[2].plot(x, snapshots[10], color="tab:red", linewidth=2, label="after 10 trees")
            for idx in step_idx:
                axes[2].annotate(
                    "",
                    xy=(x[idx], snapshots[1][idx]),
                    xytext=(x[idx], snapshots[0][idx]),
                    arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1),
                )
            axes[2].axhline(y.mean(), color="black", linewidth=1, linestyle=":")
            axes[2].set_title("arrows show the correction")

            axes[3].scatter(x, residual_final, s=9, alpha=0.35, color="tab:blue", label="residual after 10 trees")
            axes[3].axhline(0, color="black", linewidth=1)
            axes[3].set_title("after 10 trees")

            axes[0].set_ylabel("residual / correction")
            axes[2].set_ylabel("prediction")
            axes[3].set_ylabel("residual / correction")
            for axis in axes:
                axis.set_xlabel("x")
            axes[1].legend(loc="upper left")
            axes[2].legend(loc="upper left")
            fig.suptitle("Gradient boosting keeps shrinking the remaining errors", y=1.02)
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            The staircase shape is expected: shallow trees produce piecewise-constant corrections. More
            rounds create a flexible function. That flexibility is useful until the ensemble starts fitting
            noise.

            ## From residuals to classification gradients

            Binary classification usually minimizes log loss. The model maintains a score that becomes a
            probability through the logistic function. For a training row, the negative gradient is closely
            related to `observed label - current probability`:

            - a positive row predicted at `0.10` needs a large upward correction (`1 - 0.10 = 0.90`);
            - a negative row predicted at `0.90` needs a large downward correction (`0 - 0.90 = -0.90`);
            - a correct, confident row needs only a small correction.

            This is why describing boosting as “fitting mistakes” is useful, but incomplete: the precise
            mistake depends on the differentiable loss function.
            """
        ),
        code(
            """
            gradient_example = pd.DataFrame({
                "observed_y": [1, 0, 1, 0],
                "current_probability": [0.10, 0.90, 0.80, 0.15],
            })
            gradient_example["negative_gradient_y_minus_p"] = (
                gradient_example["observed_y"] - gradient_example["current_probability"]
            )
            gradient_example["interpretation"] = [
                "strong upward correction", "strong downward correction",
                "small upward correction", "small downward correction",
            ]
            gradient_example
            """
        ),
        md(
            """
            ## Three controls govern model capacity

            | Parameter | Meaning | If too small | If too large |
            |---|---|---|---|
            | `n_estimators` | number of sequential trees | underfitting | overfitting and extra compute |
            | `learning_rate` | size of each tree's contribution | needs more trees | corrections can be unstable |
            | `max_depth` | interactions each tree can represent | misses structure | fits noise and becomes harder to explain |

            `learning_rate` and `n_estimators` must be considered together. A smaller learning rate typically
            needs more trees. Shallow trees are not a weakness here; they are intentional components whose
            combined output can represent complex nonlinear structure.
            """
        ),
        code(
            """
            def make_dense_preprocessor(frame):
                categorical = frame.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
                numerical = frame.select_dtypes(include=np.number).columns.tolist()
                return ColumnTransformer([
                    ("numeric", Pipeline([
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]), numerical),
                    ("categorical", Pipeline([
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]), categorical),
                ])

            def make_model(model):
                return Pipeline([
                    ("preprocess", make_dense_preprocessor(X_dev)),
                    ("model", model),
                ])

            candidates = {
                "always no": make_model(DummyClassifier(strategy="most_frequent")),
                "logistic regression": make_model(LogisticRegression(max_iter=1200, random_state=SEED)),
                "gradient boosting": make_model(GradientBoostingClassifier(
                    n_estimators=100 if FAST_MODE else 180,
                    learning_rate=0.05, max_depth=2, random_state=SEED,
                )),
            }

            rows = []
            for name, estimator in candidates.items():
                scores = cross_validate(
                    estimator, X_dev, y_dev, cv=CV_SPLITS,
                    scoring=["accuracy", "balanced_accuracy", "f1", "neg_log_loss"], n_jobs=-1,
                )
                rows.append({
                    "model": name,
                    "accuracy": scores["test_accuracy"].mean(),
                    "balanced_accuracy": scores["test_balanced_accuracy"].mean(),
                    "f1": scores["test_f1"].mean(),
                    "log_loss": -scores["test_neg_log_loss"].mean(),
                    "accuracy_sd": scores["test_accuracy"].std(ddof=1),
                })
            cv_results = pd.DataFrame(rows).set_index("model")
            cv_results
            """
        ),
        md(
            """
            Accuracy remains easy to communicate, but it must be read beside balanced accuracy and F1: the
            majority baseline is accurate on this imbalanced dataset while finding no subscribers. Log loss
            evaluates probability quality and is also the objective optimized by the classifier.

            ## Watch training and holdout loss across boosting rounds

            Early trees reduce both training and holdout loss. Eventually, training loss may continue falling
            while holdout loss stops improving. The best iteration is selected on an internal subset of the
            development data—not the course validation set.
            """
        ),
        code(
            """
            X_fit, X_stop, y_fit, y_stop = train_test_split(
                X_dev, y_dev, test_size=0.2, stratify=y_dev, random_state=SEED
            )
            curve_preprocessor = make_dense_preprocessor(X_fit)
            X_fit_t = curve_preprocessor.fit_transform(X_fit)
            X_stop_t = curve_preprocessor.transform(X_stop)

            curve_model = GradientBoostingClassifier(
                n_estimators=220 if FAST_MODE else 400,
                learning_rate=0.05, max_depth=2, random_state=SEED,
            ).fit(X_fit_t, y_fit)

            train_loss = [log_loss(y_fit, p[:, 1]) for p in curve_model.staged_predict_proba(X_fit_t)]
            stop_loss = [log_loss(y_stop, p[:, 1]) for p in curve_model.staged_predict_proba(X_stop_t)]
            best_iteration = int(np.argmin(stop_loss) + 1)

            plt.figure(figsize=(8, 4))
            plt.plot(range(1, len(train_loss) + 1), train_loss, label="internal training")
            plt.plot(range(1, len(stop_loss) + 1), stop_loss, label="internal holdout")
            plt.axvline(best_iteration, color="black", linestyle="--",
                        label=f"best iteration = {best_iteration}")
            plt.xlabel("number of trees")
            plt.ylabel("log loss")
            plt.title("Boosting learning curve")
            plt.legend()
            plt.show()
            """
        ),
        md(
            """
            ## Refit once, then evaluate validation once

            After selecting the tree count internally, we rebuild the full pipeline on all development rows.
            Validation is used only for the final lesson-level check. The sealed test set remains untouched.
            """
        ),
        code(
            """
            final_boosting = make_model(GradientBoostingClassifier(
                n_estimators=best_iteration, learning_rate=0.05,
                max_depth=2, random_state=SEED,
            )).fit(X_dev, y_dev)
            validation_probability = final_boosting.predict_proba(X_val)[:, 1]
            validation_prediction = validation_probability >= 0.5

            validation_metrics = pd.Series({
                "accuracy": accuracy_score(y_val, validation_prediction),
                "balanced_accuracy": balanced_accuracy_score(y_val, validation_prediction),
                "f1": f1_score(y_val, validation_prediction, zero_division=0),
                "log_loss": log_loss(y_val, validation_probability),
            }, name="gradient boosting validation")
            display(validation_metrics.to_frame())
            ConfusionMatrixDisplay.from_predictions(
                y_val, validation_prediction, display_labels=["no", "yes"], colorbar=False
            )
            plt.title("Validation confusion matrix at threshold 0.5")
            plt.show()
            """
        ),
        md(
            """
            ## Gradient boosting, random forests, and CatBoost

            - A **random forest** trains many independent trees on randomized samples/features and averages
              them. Parallelism and variance reduction are central.
            - **Gradient boosting** trains dependent trees sequentially, with each tree reducing the current
              ensemble's loss. Bias reduction and careful regularization are central.
            - **CatBoost** is a gradient-boosting implementation with specialized categorical statistics and
              techniques that reduce prediction shift. Notebook 02 builds on the boosting logic established
              here before explaining CatBoost's categorical machinery.

            ## Common mistakes and leakage warnings

            - Calling each tree a complete model instead of a correction added to the ensemble.
            - Increasing tree count without considering learning rate and depth.
            - Selecting the best iteration on the final validation or test set and reporting the same score.
            - Fitting one-hot encoding on all rows before cross-validation.
            - Using accuracy alone on an imbalanced target.
            - Assuming lower training loss guarantees better generalization.

            ## Exercises

            1. Change the toy learner from `max_depth=1` to `max_depth=2`. Describe how the staircase changes.
            2. Compare `(learning_rate=0.1, n_estimators=50)` with `(0.025, 200)` using identical CV folds.
            3. Plot validation balanced accuracy, not only log loss, over boosting stages.
            4. **Challenge:** implement one boosting round for binary log loss by fitting a regression stump
               to `y - p`, then update the raw scores and probabilities.

            ## Summary

            Gradient boosting is gradient descent in function space: each shallow tree approximates a
            loss-reducing correction, and the ensemble adds those corrections gradually. Tree count, learning
            rate, and depth jointly control capacity. Internal early stopping chooses complexity; cross-
            validation estimates generalization; validation remains a final check. This foundation makes
            CatBoost's ordered categorical statistics in notebook 02 easier to understand.

            ## References

            - [scikit-learn gradient boosting user guide](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
            - [GradientBoostingClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
            - [Friedman (2001), Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451)
            - [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
            """
        ),
    ]


def n02() -> list[dict]:
    """Build notebook 02: advanced feature engineering."""
    return [
        md(
            """
            # 02 — Advanced feature engineering

            **Estimated time:** 120–150 minutes  
            **Prerequisites:** notebooks 00–01; gradient boosting, pipelines, cross-validation, and metrics.  
            **Depends on:** the pre-contact feature contract.

            ## Learning objectives

            - Build robust, leakage-safe transformations for mixed tabular data.
            - Handle sentinels, rare/unseen categories, ratios, bins, and nonlinear structure.
            - Compare one-hot encoding with CatBoost native categorical handling.
            - Use fold-level uncertainty and family-wise ablation to judge feature value.
            - Inspect the transformed feature space and connect features to model behavior.

            ## CatBoost in one minute

            **CatBoost** means *Categorical Boosting*. It is a gradient-boosted decision-tree library whose
            categorical-feature processing is part of model training rather than a separate preprocessing
            step. That matters here because fields such as `job`, `month`, `contact`, and `poutcome` are
            categories, not quantities with a meaningful numeric order.

            ### 1. Start with gradient boosting

            A single shallow decision tree captures only a small part of the pattern. Gradient boosting builds
            an ensemble sequentially:

            1. begin with a simple constant prediction;
            2. measure each row's error under the chosen loss function;
            3. fit a new tree that reduces those errors;
            4. add the new tree's contribution, scaled by the learning rate;
            5. repeat for many iterations.

            For binary classification, the ensemble produces a raw score that is converted to a probability.
            Parameters in this notebook control the process: `iterations` is the maximum number of trees,
            `depth` limits tree complexity, `learning_rate` controls the size of each correction, and
            `loss_function="Logloss"` defines what training tries to minimize.

            ### 2. Why naive categorical encoding is risky

            One-hot encoding creates one column per category. It is simple and works well for low-cardinality
            features, but the matrix becomes wide when a feature has many values. A tempting alternative is
            target encoding: replace each category with its observed positive rate. If that rate is calculated
            using the same row's target, however, the feature contains information about the answer. Rare
            categories are especially vulnerable: a category seen once would encode its label almost exactly.

            ### 3. CatBoost's ordered target statistics

            CatBoost randomly permutes the training rows and processes them as if they arrived sequentially.
            For a particular row, its category statistic is calculated only from earlier rows in that
            permutation. A smoothed prior stabilizes categories with little history.

            For example, suppose the current row has `job="admin"`. Its encoded value can use earlier
            `admin` rows and their labels, but not the current row's label or labels from later rows. CatBoost
            may also construct statistics for useful category combinations. During validation or prediction,
            statistics learned from training data are applied without using validation or future targets.

            ### 4. Ordered statistics and ordered boosting are related but different

            - **Ordered target statistics** protect categorical encodings from using a row's own target.
            - **Ordered boosting** addresses a second source of prediction shift by calculating training
              residuals with models that have not already learned from the row being evaluated.

            Both ideas use permutations, but they solve different leakage-like biases. CatBoost supports
            different boosting modes; the essential lesson is not that every configuration always uses the
            same mode, but that its categorical statistics are designed to avoid naive full-data target
            encoding.

            ### 5. How a prediction is produced

            At inference time, CatBoost receives the raw numeric and categorical columns. It applies the
            categorical statistics learned during training, sends the resulting values through every tree,
            sums the tree contributions, and converts the final score into a probability. The target is never
            required at prediction time. A threshold such as `0.5` then converts the probability into `yes`
            or `no`.

            ### 6. What CatBoost is—and is not—testing here

            Logistic regression plus one-hot encoding is our controlled feature-engineering experiment.
            CatBoost is a separate workflow benchmark combining a different encoding strategy with a more
            flexible nonlinear model. A better CatBoost result therefore does **not** prove that native
            categorical encoding alone caused the improvement.

            We use an internal development subset for early stopping, recover the selected number of trees,
            and refit on all development rows. Only then do we evaluate once on the untouched validation set.
            Training is monitored with log loss because it reacts to probability quality; final reporting uses
            accuracy, balanced accuracy, precision, recall, and F1 at the fixed `0.5` threshold.
            """
        ),
        md(
            """
            ### CatBoost workflow at a glance

            ![CatBoost workflow: raw mixed features are permuted, converted into leakage-safe ordered category statistics, processed by sequential trees, and converted into a probability and thresholded prediction.](../assets/catboost_workflow.png)

            Read the upper path from left to right. The lower inset isolates the central safety idea: naive
            target encoding calculates a category rate using every label—including the current row—whereas
            ordered statistics use only labels from earlier rows in a permutation. The illustrated numbers
            are pedagogical examples, not values calculated from the Bank Marketing dataset.
            """
        ),
        COMMON,
        code(
            """
            from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                         precision_score, recall_score, f1_score,
                                         ConfusionMatrixDisplay)
            from src.course_utils import load_bank_data, make_splits, split_xy

            development, validation, _sealed_test = make_splits(load_bank_data(), reduced=FAST_MODE)
            X_dev, y_dev = split_xy(development)
            X_val, y_val = split_xy(validation)
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
            CV_SPLITS = list(cv.split(X_dev, y_dev))
            """
        ),
        md(
            """
            ## See ordered target statistics happen row by row

            This toy example slows down CatBoost's most unusual step. Imagine that the rows below are one
            permutation. Before encoding a row, we calculate its category statistic from only the earlier
            rows with the same category. A prior supplies a stable value when little history exists.

            The left panel shows labels arriving, the middle compares leakage-safe ordered values with naive
            full-data target encoding, and the right shows how much same-category history supports each value.
            This illustrates the idea rather than reimplementing every CatBoost detail.
            """
        ),
        code(
            """
            toy_order = pd.DataFrame({
                "step": np.arange(1, 9),
                "job": ["admin", "services", "admin", "admin",
                        "services", "student", "admin", "student"],
                "subscribed": [1, 0, 0, 1, 1, 1, 0, 0],
            })
            prior, prior_weight = 0.5, 2.0
            category_sums, category_counts = {}, {}
            ordered_values, histories = [], []
            for row in toy_order.itertuples(index=False):
                previous_sum = category_sums.get(row.job, 0)
                previous_count = category_counts.get(row.job, 0)
                ordered_values.append(
                    (previous_sum + prior_weight * prior) / (previous_count + prior_weight)
                )
                histories.append(previous_count)
                category_sums[row.job] = previous_sum + row.subscribed
                category_counts[row.job] = previous_count + 1

            toy_order["ordered_stat"] = ordered_values
            toy_order["earlier_same_job_rows"] = histories
            toy_order["naive_full_data_stat"] = toy_order.groupby("job")["subscribed"].transform("mean")
            job_colors = dict(zip(toy_order["job"].unique(), sns.color_palette("Set2", 3)))

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].scatter(toy_order["step"], toy_order["subscribed"], s=110,
                            c=[job_colors[j] for j in toy_order["job"]], edgecolor="black")
            for row in toy_order.itertuples(index=False):
                axes[0].annotate(row.job, (row.step, row.subscribed), xytext=(0, 9),
                                 textcoords="offset points", ha="center", fontsize=8)
            axes[0].set(xticks=toy_order["step"], yticks=[0, 1], yticklabels=["no", "yes"],
                        xlabel="position in permutation", ylabel="observed label",
                        title="1. Rows arrive in order")

            for job, group in toy_order.groupby("job", sort=False):
                axes[1].plot(group["step"], group["ordered_stat"], marker="o", linewidth=2,
                             color=job_colors[job], label=f"{job}: ordered")
                axes[1].plot(group["step"], group["naive_full_data_stat"], linestyle="--",
                             linewidth=1.4, color=job_colors[job], alpha=0.65)
            axes[1].axhline(prior, color="black", linestyle=":", label="prior = 0.5")
            axes[1].set(xlabel="position in permutation", ylabel="encoded value", ylim=(-0.05, 1.05),
                        title="2. Encode before revealing this label")
            axes[1].legend(fontsize=7, ncol=2)

            bars = axes[2].bar(toy_order["step"], toy_order["earlier_same_job_rows"],
                               color=[job_colors[j] for j in toy_order["job"]], edgecolor="black")
            axes[2].bar_label(bars, fontsize=8)
            axes[2].set(xticks=toy_order["step"], xlabel="position in permutation",
                        ylabel="earlier rows with same job",
                        title="3. Evidence grows without peeking ahead")
            fig.suptitle("Ordered category statistics use only the past of the permutation", y=1.04)
            plt.tight_layout()
            plt.show()
            display(toy_order.round(3))
            """
        ),
        md(
            """
            ## Audit the raw feature space

            Feature hypotheses should follow evidence. We first inspect types, cardinality, missingness,
            and sentinel values. Here, `pdays=-1` is documented as "not previously contacted," so it is
            information rather than ordinary missingness.
            """
        ),
        code(
            """
            feature_audit = pd.DataFrame({
                "dtype": X_dev.dtypes.astype(str),
                "unique": X_dev.nunique(dropna=False),
                "missing_pct": X_dev.isna().mean().mul(100),
            }).sort_values(["dtype", "unique"])
            display(feature_audit)
            display(pd.crosstab(X_dev["pdays"].eq(-1), y_dev, normalize="index")
                    .rename(index={True: "never contacted", False: "contacted"}))
            """
        ),
        md(
            """
            ## Feature hypotheses before code

            - `pdays=-1` mixes a missing-history state with numeric recency. Split it into an indicator and
              nullable recency.
            - `campaign / (1 + previous)` approximates current contact pressure relative to known history.
            - `balance / age` is a deliberately weak ratio hypothesis, clipped only through the valid age
              denominator; ablation will decide whether it helps.
            - Age bins allow nonlinearity but sacrifice local detail.

            These are predictive transformations, not causal mechanisms. The transformer is row-wise and
            stateless; learned imputation, scaling, and category vocabularies remain inside the pipeline.
            """
        ),
        code(
            """
            REQUIRED_FEATURE_COLUMNS = {"pdays", "campaign", "previous", "balance", "age"}

            def add_domain_features(frame):
                \"\"\"Create pre-contact domain features using row-wise, stateless logic.\"\"\"
                missing = REQUIRED_FEATURE_COLUMNS.difference(frame.columns)
                if missing:
                    raise ValueError(f"missing required columns: {sorted(missing)}")
                if frame["age"].dropna().le(0).any():
                    raise ValueError("age must be positive when present")
                if frame["previous"].dropna().lt(0).any():
                    raise ValueError("previous must be non-negative when present")
                result = frame.copy()

                # In this dataset, pdays=-1 means the client was never previously contacted.
                result["was_previously_contacted"] = (result["pdays"] != -1).astype("int8")
                result["pdays_clean"] = result["pdays"].replace(-1, np.nan)

                # Hypotheses to validate by ablation rather than assume useful.
                result["contact_pressure"] = result["campaign"] / (1 + result["previous"])
                result["balance_per_age"] = result["balance"] / result["age"].clip(lower=18)
                result["age_band"] = pd.cut(
                    result["age"],
                    bins=[0, 29, 39, 49, 59, np.inf],
                    labels=["<30", "30s", "40s", "50s", "60+"],
                ).astype("object")

                return result.drop(columns=["pdays"])
            """
        ),
        code(
            """
            engineered_preview = add_domain_features(X_dev.head())
            engineered_preview.T
            """
        ),
        md(
            """
            ## Visualize the feature-engineering process

            Each panel corresponds to one hypothesis in `add_domain_features`: separate the `pdays=-1`
            state from genuine recency, normalize campaign effort by contact history, express balance relative
            to age, and convert continuous age into broad nonlinear regions. The plots explain what each
            transformation does; cross-validation later decides whether it helps prediction.
            """
        ),
        code(
            """
            process_sample = X_dev.sample(min(2500, len(X_dev)), random_state=SEED)
            process_features = add_domain_features(process_sample)
            fig, axes = plt.subplots(2, 2, figsize=(13, 8))

            contacted_counts = process_features["was_previously_contacted"].value_counts().reindex([0, 1])
            axes[0, 0].bar(["never contacted\\n(raw pdays = -1)", "previously contacted"],
                           contacted_counts, color=["tab:gray", "tab:blue"])
            axes[0, 0].set(ylabel="rows", title="1. Separate state from recency")
            axes[0, 0].text(0.5, 0.92, "pdays_clean retains recency only when it exists",
                            transform=axes[0, 0].transAxes, ha="center", fontsize=9)

            pressure_view = process_features.loc[:, ["campaign", "previous", "contact_pressure"]].copy()
            pressure_view["previous history"] = np.where(
                pressure_view["previous"].eq(0), "no previous contacts", "has previous contacts"
            )
            sns.scatterplot(data=pressure_view, x="campaign", y="contact_pressure",
                            hue="previous history", alpha=0.45, s=28, ax=axes[0, 1])
            axes[0, 1].set(title="2. Normalize campaign effort by history",
                           ylabel="campaign / (1 + previous)")

            lower, upper = process_features["balance_per_age"].quantile([0.01, 0.99])
            ratio_view = process_features["balance_per_age"].clip(lower, upper)
            axes[1, 0].scatter(process_sample["age"], ratio_view, s=12, alpha=0.25,
                               color="tab:purple")
            axes[1, 0].axhline(0, color="black", linewidth=1)
            axes[1, 0].set(xlabel="age", ylabel="balance / age (1st–99th pct shown)",
                           title="3. Express balance relative to age")

            age_counts = process_features["age_band"].value_counts().reindex(
                ["<30", "30s", "40s", "50s", "60+"]
            )
            axes[1, 1].bar(age_counts.index, age_counts.values, color=sns.color_palette("Blues", 5))
            axes[1, 1].set(xlabel="engineered age band", ylabel="rows",
                           title="4. Turn continuous age into regions")

            fig.suptitle("Raw columns become explicit, testable feature hypotheses", y=1.01)
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)` groups categories that
            are rare in each training fold and safely handles unseen validation values. It returns a sparse
            matrix, which logistic regression and LightGBM can consume. Avoid forcing dense output: the
            memory cost can grow rapidly with cardinality.

            ## Choosing interpretable classification metrics

            Accuracy is easy to explain: *what fraction of predictions were correct?* But the positive
            class is uncommon, so an always-negative classifier can look strong. We therefore report
            accuracy as the headline metric and keep four diagnostic metrics:

            - **balanced accuracy:** average recall across the positive and negative classes;
            - **precision:** among predicted subscriptions, how many actually subscribed;
            - **recall:** among actual subscriptions, how many the model found;
            - **F1:** a compact balance of precision and recall.

            These label metrics require a threshold; this notebook uses `0.5` consistently. Changing the
            threshold changes all five values, so it must be chosen from business costs rather than adjusted
            after viewing validation results.
            """
        ),
        code(
            """
            positive_rate = y_dev.mean()
            majority_accuracy = max(positive_rate, 1 - positive_rate)
            print(f"positive class: {positive_rate:.1%}")
            print(f"always predict 'no' accuracy: {majority_accuracy:.1%}")
            """
        ),
        md(
            """
            ## Build the preprocessing pipeline

            Numeric columns are median-imputed and standardized for logistic regression. Categorical
            columns are imputed, one-hot encoded, and protected against rare or unseen values. Defining the
            complete preprocessor here makes the fit boundary visible: every learned statistic is fitted
            only on a training fold by the enclosing pipeline.
            """
        ),
        code(
            """
            def make_preprocessor(frame):
                features = frame.drop(columns=[TARGET], errors="ignore")
                categorical = features.select_dtypes(
                    include=["object", "category", "bool"]
                ).columns.tolist()
                numerical = features.select_dtypes(include=np.number).columns.tolist()

                numeric_pipe = Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ])
                categorical_pipe = Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=10,
                        sparse_output=True,
                    )),
                ])
                return ColumnTransformer([
                    ("numeric", numeric_pipe, numerical),
                    ("categorical", categorical_pipe, categorical),
                ], sparse_threshold=0.3)
            """
        ),
        code(
            """
            baseline = Pipeline([
                ("preprocess", make_preprocessor(development)),
                ("model", LogisticRegression(max_iter=1200, random_state=SEED)),
            ])
            engineered_frame = add_domain_features(development)
            engineered = Pipeline([
                ("features", FunctionTransformer(add_domain_features, validate=False)),
                ("preprocess", make_preprocessor(engineered_frame)),
                ("model", LogisticRegression(max_iter=1200, random_state=SEED)),
            ])

            fold_metrics = {}

            def cv_summary(name, estimator):
                scores = cross_validate(
                    estimator, X_dev, y_dev, cv=CV_SPLITS,
                    scoring=["accuracy", "balanced_accuracy", "precision", "recall", "f1"],
                    n_jobs=-1,
                )
                fold_metrics[name] = pd.DataFrame({
                    metric: scores[f"test_{metric}"]
                    for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
                }).rename_axis("fold")
                accuracy = scores["test_accuracy"]
                return {
                    "experiment": name,
                    "accuracy_mean": accuracy.mean(),
                    "accuracy_sd": accuracy.std(ddof=1),
                    "accuracy_se": accuracy.std(ddof=1) / np.sqrt(len(accuracy)),
                    "balanced_accuracy": scores["test_balanced_accuracy"].mean(),
                    "precision": scores["test_precision"].mean(),
                    "recall": scores["test_recall"].mean(),
                    "f1": scores["test_f1"].mean(),
                }

            ablation = pd.DataFrame([cv_summary("raw features", baseline),
                                     cv_summary("domain features", engineered)]).set_index("experiment")
            ablation
            """
        ),
        md(
            """
            The standard deviation describes variation among these folds; the standard error describes
            uncertainty in their mean under the fold-sampling approximation. With only 3–5 correlated
            folds, neither is a formal confidence guarantee. Treat tiny accuracy differences as inconclusive and
            consider operational cost as well as the point estimate.

            ## Ablate feature families, not individual columns

            The engineered set contains three hypotheses: contact history, contact pressure, and customer
            demographics. Removing one family at a time makes the comparison interpretable while keeping
            all learned preprocessing inside each CV fold.
            """
        ),
        code(
            """
            FEATURE_FAMILIES = {
                "contact history": ["was_previously_contacted", "pdays_clean"],
                "contact pressure": ["contact_pressure"],
                "demographic": ["balance_per_age", "age_band"],
            }

            def domain_features_without(frame, excluded=()):
                return add_domain_features(frame).drop(columns=list(excluded), errors="ignore")

            def pipeline_without(excluded=()):
                schema = domain_features_without(development, excluded)
                transform = FunctionTransformer(
                    domain_features_without, kw_args={"excluded": tuple(excluded)}, validate=False
                )
                return Pipeline([
                    ("features", transform),
                    ("preprocess", make_preprocessor(schema)),
                    ("model", LogisticRegression(max_iter=1200, random_state=SEED)),
                ])

            family_ablation = [cv_summary("all engineered", pipeline_without())]
            for family, columns in FEATURE_FAMILIES.items():
                family_ablation.append(cv_summary(f"without {family}", pipeline_without(columns)))
            family_ablation = pd.DataFrame(family_ablation).set_index("experiment")
            family_ablation["delta_accuracy"] = (
                family_ablation["accuracy_mean"] - family_ablation.loc["all engineered", "accuracy_mean"]
            )
            family_ablation["delta_balanced_accuracy"] = (
                family_ablation["balanced_accuracy"]
                - family_ablation.loc["all engineered", "balanced_accuracy"]
            )
            display(family_ablation.sort_values("accuracy_mean", ascending=False))

            all_folds = fold_metrics["all engineered"]
            paired_rows = []
            for experiment in family_ablation.index.drop("all engineered"):
                difference = fold_metrics[experiment] - all_folds
                paired_rows.append({
                    "experiment": experiment,
                    **{f"delta_{metric}": difference[metric].mean()
                       for metric in ["accuracy", "balanced_accuracy", "recall", "f1"]},
                })
            paired_differences = pd.DataFrame(paired_rows).set_index("experiment")
            display(paired_differences)

            plot_data = pd.concat(
                {name: values[["accuracy", "balanced_accuracy"]]
                 for name, values in fold_metrics.items()
                 if name == "all engineered" or name.startswith("without")},
                names=["experiment", "fold"],
            ).reset_index().melt(
                id_vars=["experiment", "fold"], var_name="metric", value_name="score"
            )
            g = sns.catplot(
                data=plot_data, x="score", y="experiment", hue="fold", col="metric",
                kind="point", sharex=False, height=3.5, aspect=1.15,
            )
            g.set_titles("{col_name}")
            g.fig.suptitle("Paired fold results: engineered feature-family ablation", y=1.05)
            plt.show()
            """
        ),
        md(
            """
            ## Feature validation

            Feature code deserves tests. We check row preservation, finite constructed ratios, valid age
            bands, and the no-current-duration contract. Production validation would also enforce allowed
            categories, ranges, and data types at ingestion.
            """
        ),
        code(
            """
            check = add_domain_features(X_dev)
            assert len(check) == len(X_dev)
            assert check.index.equals(X_dev.index)
            assert "duration" not in check
            assert np.isfinite(check["contact_pressure"]).all()
            assert np.isfinite(check["balance_per_age"]).all()
            assert check["age_band"].notna().all()
            original = X_dev.head(2).copy(deep=True)
            add_domain_features(original)
            pd.testing.assert_frame_equal(original, X_dev.head(2))

            empty = add_domain_features(X_dev.iloc[:0])
            assert empty.empty and list(empty.columns) == list(check.columns)

            with_null = X_dev.head(1).copy()
            with_null.loc[with_null.index[0], "age"] = np.nan
            null_result = add_domain_features(with_null)
            assert pd.isna(null_result["balance_per_age"].iloc[0])
            assert pd.isna(null_result["age_band"].iloc[0])

            try:
                add_domain_features(X_dev.drop(columns="age"))
                raise AssertionError("missing-column validation did not run")
            except ValueError as error:
                assert "age" in str(error)

            invalid_age = X_dev.head(1).copy()
            invalid_age["age"] = 0
            try:
                add_domain_features(invalid_age)
                raise AssertionError("age validation did not run")
            except ValueError as error:
                assert "positive" in str(error)

            print("feature checks passed, including schema, null, empty, and mutation cases")
            """
        ),
        md(
            """
            ## Inspect what the pipeline actually learned

            Human-readable output names catch accidental omissions and category explosions. Coefficients
            below are associations conditional on the other transformed inputs, not causal effects.
            Because numeric inputs are standardized, their magnitudes are more comparable; one-hot
            coefficients require extra care: all category indicators are retained, so there is no single
            omitted reference category and correlated columns can make individual coefficients unstable.
            Use them to inspect model behavior, not as unique causal effects.
            """
        ),
        code(
            """
            inspection_model = engineered.fit(X_dev, y_dev)
            names = inspection_model.named_steps["preprocess"].get_feature_names_out()
            coefficients = pd.Series(
                inspection_model.named_steps["model"].coef_[0], index=names, name="coefficient"
            )
            display(pd.concat([coefficients.nsmallest(10), coefficients.nlargest(10)]).to_frame())
            print(f"{len(X_dev.columns)} raw columns -> {len(names)} transformed columns")

            unseen = X_val.head(1).copy()
            unseen["job"] = "__NEW_JOB_AT_INFERENCE__"
            transformed_unseen = inspection_model[:-1].transform(unseen)
            assert transformed_unseen.shape[1] == len(names)
            print("unseen category transformed safely; output shape", transformed_unseen.shape)
            """
        ),
        md(
            """
            ## Native categorical handling with CatBoost

            CatBoost builds statistics for categorical values using ordered target statistics designed to
            reduce target leakage. We pass raw strings and categorical column indices; we do **not** fit a
            target encoder on the whole dataset. Missing values would need explicit string/category handling.
            A stratified subset of development data is used for early stopping. The course validation set
            remains untouched until the final comparison. Log loss is used for early stopping because it is
            sensitive to probability quality even when the predicted class does not change; accuracy remains
            the headline reporting metric.
            """
        ),
        code(
            """
            from catboost import CatBoostClassifier
            cat_columns = X_dev.select_dtypes(include="object").columns.tolist()
            X_dev_cat, X_val_cat = X_dev.copy(), X_val.copy()
            for c in cat_columns:
                X_dev_cat[c] = X_dev_cat[c].fillna("__MISSING__")
                X_val_cat[c] = X_val_cat[c].fillna("__MISSING__")

            X_cat_fit, X_cat_stop, y_cat_fit, y_cat_stop = train_test_split(
                X_dev_cat, y_dev, test_size=0.15, stratify=y_dev, random_state=SEED
            )

            cat_stop_model = CatBoostClassifier(
                iterations=180 if FAST_MODE else 450, depth=6, learning_rate=0.06,
                loss_function="Logloss", eval_metric="Logloss", random_seed=SEED,
                verbose=False, allow_writing_files=False,
            )
            cat_stop_model.fit(X_cat_fit, y_cat_fit, cat_features=cat_columns,
                               eval_set=(X_cat_stop, y_cat_stop),
                               early_stopping_rounds=40, verbose=False)
            best_iterations = cat_stop_model.get_best_iteration() + 1
            cat_model = CatBoostClassifier(
                iterations=best_iterations, depth=6, learning_rate=0.06,
                loss_function="Logloss", random_seed=SEED,
                verbose=False, allow_writing_files=False,
            )
            cat_model.fit(X_dev_cat, y_dev, cat_features=cat_columns, verbose=False)
            print(f"refit CatBoost on all development rows with {best_iterations} trees")

            onehot_model = inspection_model
            raw_onehot_model = baseline.fit(X_dev, y_dev)
            def threshold_metrics(y_true, probability, threshold=0.5):
                prediction = probability >= threshold
                return {
                    "accuracy": accuracy_score(y_true, prediction),
                    "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
                    "precision": precision_score(y_true, prediction, zero_division=0),
                    "recall": recall_score(y_true, prediction, zero_division=0),
                    "f1": f1_score(y_true, prediction, zero_division=0),
                }

            validation_probabilities = {
                "always no": np.zeros(len(y_val)),
                "raw one-hot logistic": raw_onehot_model.predict_proba(X_val)[:, 1],
                "engineered one-hot logistic": onehot_model.predict_proba(X_val)[:, 1],
                "native CatBoost": cat_model.predict_proba(X_val_cat)[:, 1],
            }
            comparison = pd.DataFrame({
                name: threshold_metrics(y_val, probability)
                for name, probability in validation_probabilities.items()
            }).T
            display(comparison)

            fig, axes = plt.subplots(1, 4, figsize=(17, 3.5))
            for axis, (name, probability) in zip(axes, validation_probabilities.items()):
                ConfusionMatrixDisplay.from_predictions(
                    y_val, probability >= 0.5, display_labels=["no", "yes"],
                    colorbar=False, ax=axis,
                )
                axis.set_title(name)
            fig.suptitle("Validation confusion matrices at threshold 0.5", y=1.03)
            plt.tight_layout()
            plt.show()

            best_name = comparison["balanced_accuracy"].idxmax()
            print(
                f"At threshold 0.5, {best_name} has the highest balanced accuracy "
                f"({comparison.loc[best_name, 'balanced_accuracy']:.3f}), with "
                f"accuracy={comparison.loc[best_name, 'accuracy']:.3f}, "
                f"precision={comparison.loc[best_name, 'precision']:.3f}, and "
                f"recall={comparison.loc[best_name, 'recall']:.3f}."
            )
            """
        ),
        md(
            """
            Native categorical support is not automatically superior: it changes the model family as well
            as the encoding. A clean encoding comparison would hold the estimator and tuning budget fixed,
            which is not fully possible here. Treat this as a workflow comparison.

            ## What this experiment tells us

            In the stored reduced-mode run, domain features do not improve overall accuracy, but they
            slightly improve recall, F1, and balanced accuracy. The fold-level changes are tiny, so the
            evidence is insufficient to claim that the added feature complexity is worthwhile. CatBoost
            produces the strongest validation balanced accuracy and recall at threshold `0.5`, while the
            confusion matrices show that every model still misses most subscribers. That is a threshold and
            decision-policy question for a later lesson, not a reason to tune on this validation set here.

            ## Common mistakes and leakage warnings

            - Computing target means or category frequencies on all rows before CV.
            - Creating bins using full-data quantiles outside the pipeline.
            - Treating `pdays=-1` as negative recency.
            - Densifying a large sparse one-hot matrix without checking memory.
            - Keeping engineered features because they sound plausible despite null ablation results.

            ## Exercises

            1. Add cyclical month encoding and explain why chronology is still incomplete.
            2. Replace fixed age bands with a spline (`SplineTransformer`) and compare it by CV.
            3. Plot fold-level accuracy and balanced-accuracy changes for each feature-family ablation.
            4. **Challenge:** implement a scikit-learn-compatible transformer that learns rare categories
               during `fit` and maps them during `transform`; verify it with estimator checks and an unseen category.

            ## Summary

            Feature engineering is safe when the prediction contract is explicit, row-wise logic is pure,
            and every learned transformation is fit inside CV. Ablation—not narrative plausibility—decides
            whether a feature earns operational complexity.

            ## References

            - [ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html#column-transformer-for-heterogeneous-data)
            - [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
            - [Pipeline and composite estimators](https://scikit-learn.org/stable/modules/compose.html)
            - [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
            - [Feature engineering chapter, scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/python_scripts/03_categorical_pipeline.html)
            - [CatBoost categorical features](https://catboost.ai/en/docs/features/categorical-features)
            - [CatBoost training algorithm](https://catboost.ai/docs/en/concepts/algorithm-main-stages)
            - [How CatBoost transforms categorical features](https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic)
            - [CatBoost paper: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516)
            """
        ),
    ]


def n03() -> list[dict]:
    """Build notebook 03: imbalanced learning."""
    return [
        md(
            """
            # 03 — Imbalanced learning

            **Estimated time:** 100–140 minutes  
            **Prerequisites:** notebooks 00–02; cross-validation, metrics, and pipelines.  
            **Depends on:** the split contract from notebook 00 and feature pipeline from notebook 02.

            ## Learning objectives

            - Compare class weights, thresholds, under/oversampling, SMOTE, and SMOTENC.
            - Resample only within CV training folds using `imblearn.pipeline.Pipeline`.
            - Evaluate classification behavior, probability quality, and business cost separately.
            - Recognize when synthetic interpolation is semantically invalid.
            """
        ),
        COMMON,
        code(
            """
            from sklearn.base import clone
            from sklearn.dummy import DummyClassifier
            from sklearn.metrics import make_scorer
            from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline as SkPipeline
            from imblearn.pipeline import Pipeline as ImbPipeline
            from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
            from imblearn.under_sampling import RandomUnderSampler
            from src.course_utils import load_bank_data, make_splits, make_preprocessor, split_xy

            development, validation, _sealed_test = make_splits(load_bank_data(), reduced=FAST_MODE)
            X_dev, y_dev = split_xy(development)
            X_val, y_val = split_xy(validation)
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
            class_audit = pd.DataFrame({
                "development": y_dev.value_counts().sort_index(),
                "validation": y_val.value_counts().sort_index(),
            }).rename(index={0: "no", 1: "yes"})
            class_audit.loc[:, "development_rate"] = class_audit["development"] / len(y_dev)
            class_audit.loc[:, "validation_rate"] = class_audit["validation"] / len(y_val)
            display(class_audit)
            print(f"Positive-class prevalence: {y_dev.mean():.4f}")
            """
        ),
        code(
            """
            class_plot = class_audit[["development", "validation"]].T
            ax = class_plot.plot(kind="bar", stacked=True, figsize=(8, 4),
                                 color=["#4C78A8", "#E45756"])
            ax.set(title="The positive class is rare in both modeling splits",
                   xlabel="Split", ylabel="Number of observations")
            ax.tick_params(axis="x", rotation=0)
            for container in ax.containers:
                ax.bar_label(container, label_type="center", color="white", fontsize=9)
            ax.legend(title="Outcome", labels=["No", "Yes"])
            plt.tight_layout()
            """
        ),
        md(
            """
            ## Why sampler placement matters

            Resampling before cross-validation lets synthetic or duplicated observations influence both
            training and validation folds. That is leakage. An imbalanced-learn pipeline calls the sampler
            only during `fit`, so each fold's validation rows remain untouched.

            SMOTE interpolates between minority neighbors. Applied after one-hot encoding, it can create
            fractional pseudo-categories. That may run numerically but is semantically dubious. We include it
            as a cautionary comparison and use SMOTENC for mixed data.
            """
        ),
        code(
            """
            preprocessors = {name: make_preprocessor(development) for name in
                             ["dummy", "plain", "weighted", "under", "over", "smote"]}
            candidates = {
                "dummy_prior": SkPipeline([("preprocess", preprocessors["dummy"]),
                                            ("model", DummyClassifier(strategy="prior"))]),
                "plain": SkPipeline([("preprocess", preprocessors["plain"]),
                                     ("model", LogisticRegression(max_iter=1200, random_state=SEED))]),
                "class_weight": SkPipeline([("preprocess", preprocessors["weighted"]),
                                            ("model", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=SEED))]),
                "random_under": ImbPipeline([("preprocess", preprocessors["under"]),
                                             ("sample", RandomUnderSampler(random_state=SEED)),
                                             ("model", LogisticRegression(max_iter=1200, random_state=SEED))]),
                "random_over": ImbPipeline([("preprocess", preprocessors["over"]),
                                            ("sample", RandomOverSampler(random_state=SEED)),
                                            ("model", LogisticRegression(max_iter=1200, random_state=SEED))]),
                "SMOTE_after_onehot_caution": ImbPipeline([("preprocess", preprocessors["smote"]),
                                                           ("sample", SMOTE(random_state=SEED)),
                                                           ("model", LogisticRegression(max_iter=1200, random_state=SEED))]),
            }
            """
        ),
        code(
            """
            scoring = {
                "balanced_accuracy": "balanced_accuracy",
                "precision": make_scorer(precision_score, zero_division=0),
                "recall": "recall",
                "f1": "f1",
                "log_loss": "neg_log_loss",
            }
            cv_rows = []
            for name, estimator in candidates.items():
                scores = cross_validate(estimator, X_dev, y_dev, cv=cv, scoring=scoring, n_jobs=-1)
                row = {"method": name}
                for metric in scoring:
                    values = scores[f"test_{metric}"] * (-1 if metric == "log_loss" else 1)
                    row[f"{metric}_mean"] = values.mean()
                    row[f"{metric}_std"] = values.std(ddof=1)
                cv_rows.append(row)
            cv_comparison = pd.DataFrame(cv_rows).set_index("method")
            cv_comparison.sort_values("balanced_accuracy_mean", ascending=False).round(4)
            """
        ),
        md(
            """
            Resampling changes the class prior seen during fitting, so raw probabilities can become badly
            calibrated even when recall improves. Class weighting has a similar effect on the loss. Threshold
            metrics must therefore be read alongside log loss and calibration.

            ## SMOTENC: preserve categorical semantics

            The preprocessing below outputs numeric columns first and ordinal-encoded categories second.
            `SMOTENC` is told exactly which intermediate columns are categorical. After sampling, numeric
            columns are scaled and categorical codes are one-hot encoded. This last step matters: ordinal
            codes are identifiers, so feeding the codes directly to logistic regression would impose a false
            distance and ordering between categories.
            """
        ),
        code(
            """
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import OrdinalEncoder, StandardScaler
            from src.course_utils import feature_groups

            numerical, categorical = feature_groups(development)
            ordinal_preprocess = ColumnTransformer([
                ("numeric", SimpleImputer(strategy="median"), numerical),
                ("categorical", SkPipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]), categorical),
            ], sparse_threshold=0)
            numerical_indices = list(range(len(numerical)))
            categorical_indices = list(range(len(numerical), len(numerical) + len(categorical)))
            post_smotenc = ColumnTransformer([
                ("numeric", StandardScaler(), numerical_indices),
                ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 categorical_indices),
            ], sparse_threshold=0)
            smotenc_model = ImbPipeline([
                ("preprocess", ordinal_preprocess),
                ("sample", SMOTENC(categorical_features=categorical_indices, random_state=SEED)),
                ("postprocess", post_smotenc),
                ("model", LogisticRegression(max_iter=1200, random_state=SEED)),
            ])
            scores = cross_validate(smotenc_model, X_dev, y_dev, cv=cv, scoring=scoring, n_jobs=-1)
            smotenc_row = {}
            for metric in scoring:
                values = scores[f"test_{metric}"] * (-1 if metric == "log_loss" else 1)
                smotenc_row[f"{metric}_mean"] = values.mean()
                smotenc_row[f"{metric}_std"] = values.std(ddof=1)
            cv_comparison.loc["SMOTENC"] = smotenc_row
            cv_comparison.sort_values("balanced_accuracy_mean", ascending=False).round(4)
            """
        ),
        md(
            """
            ### Visual comparison across cross-validation folds

            A method can gain recall simply by predicting the positive class much more often. The panels
            below make the resulting precision–recall trade-off visible. Error bars show one standard
            deviation across folds; they describe split sensitivity rather than a confidence interval.
            """
        ),
        code(
            """
            plot_order = cv_comparison.sort_values(
                "balanced_accuracy_mean", ascending=True
            ).index
            metrics_to_plot = ["balanced_accuracy", "precision", "recall", "f1"]
            titles = ["Balanced accuracy", "Precision", "Recall", "F1 score"]
            fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
            colors = sns.color_palette("colorblind", n_colors=len(plot_order))
            for ax, metric, title in zip(axes.flat, metrics_to_plot, titles):
                means = cv_comparison.loc[plot_order, f"{metric}_mean"]
                errors = cv_comparison.loc[plot_order, f"{metric}_std"]
                ax.barh(plot_order, means, xerr=errors, color=colors,
                        alpha=.9, capsize=3)
                ax.set(title=title, xlabel="CV mean ± 1 fold SD", xlim=(0, 1))
                ax.axvline(.5, color="grey", linestyle=":", linewidth=1)
            fig.suptitle("Resampling changes the operating trade-off", fontsize=14, y=1.01)
            plt.tight_layout()
            """
        ),
        md("## Validation behavior and threshold adjustment"),
        code(
            """
            from src.course_utils import threshold_table

            def operational_metrics(y_true, probability, threshold=0.5):
                # Return threshold and probability metrics without area-under-curve scores.
                prediction = np.asarray(probability) >= threshold
                tn, fp, fn, tp = confusion_matrix(
                    y_true, prediction, labels=[0, 1]
                ).ravel()
                return {
                    "log_loss": log_loss(y_true, probability),
                    "precision": precision_score(y_true, prediction, zero_division=0),
                    "recall": recall_score(y_true, prediction, zero_division=0),
                    "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
                    "cost": float(fp + 5 * fn),
                }

            # Choose the threshold from out-of-fold development predictions. The validation
            # set remains independent evidence rather than being used both to tune and report.
            oof_probability = cross_val_predict(
                clone(candidates["plain"]), X_dev, y_dev, cv=cv,
                method="predict_proba", n_jobs=-1,
            )[:, 1]
            threshold_candidates = threshold_table(y_dev, oof_probability)
            selected = threshold_candidates.sort_values(["cost", "threshold"]).iloc[0]
            selected_threshold = float(selected["threshold"])
            print(f"OOF development cost-selected threshold: {selected_threshold:.2f}")

            validation_rows = []
            fitted = {}
            for name in ["dummy_prior", "plain", "class_weight", "random_over"]:
                fitted[name] = candidates[name].fit(X_dev, y_dev)
                probability = fitted[name].predict_proba(X_val)[:, 1]
                validation_rows.append({"method": name, **operational_metrics(y_val, probability)})
            validation_comparison = pd.DataFrame(validation_rows).set_index("method")
            display(validation_comparison.round(4))

            plain_probability = fitted["plain"].predict_proba(X_val)[:, 1]
            display(pd.DataFrame({
                "plain@0.5": operational_metrics(y_val, plain_probability, .5),
                "plain@OOF-selected": operational_metrics(
                    y_val, plain_probability, selected_threshold),
            }).T.round(4))
            """
        ),
        code(
            """
            validation_predictions = {
                "Default threshold (0.50)": plain_probability >= 0.50,
                f"OOF-selected threshold ({selected_threshold:.2f})":
                    plain_probability >= selected_threshold,
            }
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            vmax = max(confusion_matrix(y_val, pred, labels=[0, 1]).max()
                       for pred in validation_predictions.values())
            for ax, (title, prediction) in zip(axes, validation_predictions.items()):
                matrix = confusion_matrix(y_val, prediction, labels=[0, 1])
                sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                            vmin=0, vmax=vmax, square=True, ax=ax,
                            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
                ax.set(title=title, xlabel="Predicted outcome", ylabel="Actual outcome")
            fig.suptitle("Lowering the threshold recovers positives but creates more false alarms",
                         fontsize=13, y=1.03)
            plt.tight_layout()
            """
        ),
        md(
            """
            ### Reading the threshold result correctly

            The false-negative cost of 5 and false-positive cost of 1 are teaching assumptions, not facts
            learned from the dataset. In a real deployment, replace them with validated economic or clinical
            consequences and include capacity constraints. The OOF-selected threshold is evaluated once on
            validation; repeatedly revisiting it after seeing validation performance would gradually turn the
            validation set into training data.
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            axes[0].plot(threshold_candidates["threshold"], threshold_candidates["precision"],
                         label="precision")
            axes[0].plot(threshold_candidates["threshold"], threshold_candidates["recall"],
                         label="recall")
            axes[0].axvline(selected_threshold, color="tab:red", linestyle="--",
                            label=f"selected = {selected_threshold:.2f}")
            axes[0].legend()
            axes[0].set(xlabel="Decision threshold", ylabel="Score",
                        title="OOF precision and recall by threshold")

            axes[1].plot(threshold_candidates["threshold"], threshold_candidates["cost"])
            axes[1].axvline(selected_threshold, color="tab:red", linestyle="--",
                            label=f"selected = {selected_threshold:.2f}")
            axes[1].set(xlabel="Decision threshold", ylabel="OOF development cost",
                        title="Threshold selection uses OOF predictions")
            axes[1].legend()
            plt.tight_layout()
            """
        ),
        code(
            """
            from sklearn.calibration import CalibrationDisplay
            fig, ax = plt.subplots(figsize=(6, 5))
            for name in ["plain", "class_weight", "random_over"]:
                p = fitted[name].predict_proba(X_val)[:, 1]
                CalibrationDisplay.from_predictions(y_val, p, n_bins=8, strategy="quantile", name=name, ax=ax)
            ax.set_title("Resampling/weighting can distort probabilities")
            plt.tight_layout()
            """
        ),
        md(
            """
            ### What this experiment establishes—and what it does not

            - **Classification:** precision, recall, F1, and balanced accuracy depend on a threshold. They
              expose the operational trade-off directly and should be interpreted with class prevalence.
            - **Probability quality:** log loss and calibration diagnose whether probabilities can support
              expected-value decisions. Resampling can improve recall while harming probability quality.
            - **Uncertainty:** fold standard deviations describe split sensitivity, not a formal confidence
              interval. Repeated or nested CV is appropriate when selection uncertainty matters.
            - **Scope:** this is predictive evaluation under an i.i.d. stratified split. It does not estimate
              the causal effect of calling a customer, and the source data cannot support grouped or temporal
              validation because stable customer IDs and complete timestamps are absent.

            For deployment, log the dataset hash, split seed, package versions, candidate parameters, fold-level
            scores, selected threshold, cost assumptions, and the serialized end-to-end pipeline. Monitor both
            score calibration and the operational precision/recall trade-off as prevalence changes.
            """
        ),
        md(
            """
            **When not to synthesize:** avoid SMOTE when neighborhoods are not meaningful, minority data
            contain label noise, constraints can be violated, categories have high cardinality, temporal
            order matters, or calibrated probabilities are central and recalibration data are scarce.
            Threshold adjustment is often the simplest operational intervention when ranking is already good.

            ## Common mistakes and leakage warnings

            - Resampling once before CV or before the train/validation split.
            - Applying ordinary SMOTE to raw ordinal category codes.
            - Comparing recall at 0.5 while ignoring precision and calibration.
            - Assuming the balanced training distribution is the deployment prevalence.
            - Tuning sampling ratio and threshold on the test set.

            ## Exercises

            1. Vary `sampling_strategy` and plot balanced accuracy versus log loss with fold-level uncertainty.
            2. Calibrate the class-weighted model using nested CV or a dedicated calibration split; compare
               reliability curves, log loss, and Brier score before and after calibration.
            3. Add bootstrap confidence intervals for validation recall, precision, and business cost.
            4. **Challenge:** design a repeated nested-CV experiment comparing threshold tuning with SMOTENC,
               including uncertainty and a fixed business cost.

            ## Summary

            Class weighting and resampling alter the fitting objective; threshold adjustment alters only the
            decision rule. None is automatically best. Samplers belong inside CV pipelines, and mixed data
            require categorical-aware synthesis when synthesis is justified at all.

            ## References

            - [imbalanced-learn pipeline](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html)
            - [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
            - [SMOTENC](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html)
            """
        ),
    ]


def n04() -> list[dict]:
    """Build notebook 04: Optuna hyperparameter optimization."""
    return [
        md(
            """
            # 04 — Optuna hyperparameter optimization

            **Estimated time:** 120–160 minutes  
            **Prerequisites:** notebooks 00–03; gradient boosting, CV, and early stopping.  
            **Depends on:** the development/validation/test contract from notebook 00.

            ## Learning objectives

            - Translate model-development goals into an Optuna objective.
            - Design a bounded, reproducible search space for CatBoost.
            - Combine cross-validation, early stopping, and Optuna pruning without leakage.
            - Inspect trials, parameter importance, and optimization history.
            - Promote one tuned candidate to validation without repeatedly optimizing validation.
            """
        ),
        COMMON_NO_LGBM_WARNING,
        code(
            """
            from catboost import CatBoostClassifier
            from sklearn.model_selection import StratifiedKFold
            from sklearn.metrics import log_loss
            from src.course_utils import (classification_metrics, load_bank_data, make_preprocessor,
                                          make_splits, split_xy)

            development, validation, _sealed_test = make_splits(load_bank_data(), reduced=FAST_MODE)
            X_dev, y_dev = split_xy(development)
            X_val, y_val = split_xy(validation)
            import catboost
            print("Library versions:", catboost.__version__)
            """
        ),
        md(
            """
            ## What Optuna is responsible for

            Optuna is not a magic model improver. It is a controlled experiment runner for model-development
            choices: propose parameters, fit a candidate under the same data boundary, observe a metric, and
            use the trial history to choose the next candidate.

            In this notebook the model is CatBoost, but the center of gravity is Optuna:

            - the objective returns cross-validated log loss from development data only;
            - each fold owns its own preprocessing and early-stopping split;
            - the search space is intentionally bounded because search budget is limited;
            - pruning can stop weak trials after partial evidence;
            - validation is used once after the study to compare the tuned candidate with the baseline.
            """
        ),
        code(
            """
            # A single untuned baseline gives Optuna something honest to beat.
            preprocessor = make_preprocessor(development, scale_numeric=False)
            X_dev_encoded = preprocessor.fit_transform(X_dev, y_dev)
            X_val_encoded = preprocessor.transform(X_val)

            catboost_baseline = CatBoostClassifier(
                loss_function="Logloss", eval_metric="Logloss", iterations=1000,
                learning_rate=0.04, depth=6, min_data_in_leaf=30,
                l2_leaf_reg=3.0, random_seed=SEED, thread_count=-1,
                allow_writing_files=False, verbose=False,
            )
            catboost_baseline.fit(
                X_dev_encoded, y_dev, eval_set=[(X_val_encoded, y_val)],
                use_best_model=True, early_stopping_rounds=50, verbose=False,
            )
            cat_p = catboost_baseline.predict_proba(X_val_encoded)[:, 1]
            print("CatBoost best iteration:", catboost_baseline.get_best_iteration())
            pd.Series(classification_metrics(y_val, cat_p), name="untuned CatBoost")
            """
        ),
        md(
            """
            ## Design the Optuna study

            The objective below uses only development folds. Each fold fits its own encoder and uses its
            validation fold for early stopping. The trial reports intermediate mean log loss after each fold,
            enabling Optuna's median pruner. The sealed test set is absent from the function, and the
            validation set is also absent from the objective.

            `FAST_MODE=1` runs 8 trials × 3 folds; full mode runs 25 × 5. This is an educational search,
            not evidence that 25 trials fully explores the space.

            ### Search-space choices

            | Parameter | Why Optuna tunes it | Bound used here |
            |---|---|---|
            | `learning_rate` | shrinkage/iteration tradeoff | log scale, 0.02–0.12 |
            | `depth` | tree depth and interaction complexity | 4–8 |
            | `min_data_in_leaf` | leaf regularization | 10–80 |
            | `subsample` | row sampling regularization | 0.70–1.00 |
            | `rsm` | feature sampling regularization | 0.70–1.00 |
            | `l2_leaf_reg`, `random_strength` | regularization and split-score noise | log-scaled positive ranges |
            """
        ),
        code(
            """
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

            def objective(trial):
                # Return development-only cross-validated log loss for one Optuna trial.
                params = {
                    "loss_function": "Logloss", "eval_metric": "Logloss",
                    "thread_count": -1, "random_seed": SEED, "iterations": 800,
                    "allow_writing_files": False, "verbose": False,
                    "bootstrap_type": "Bernoulli",
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                    "depth": trial.suggest_int("depth", 4, 8),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 80),
                    "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                    "rsm": trial.suggest_float("rsm", 0.7, 1.0),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
                    "random_strength": trial.suggest_float("random_strength", 1e-3, 5.0, log=True),
                }
                fold_scores = []
                for fold, (fit_idx, stop_idx) in enumerate(cv.split(X_dev, y_dev)):
                    X_fit, X_stop = X_dev.iloc[fit_idx], X_dev.iloc[stop_idx]
                    y_fit, y_stop = y_dev.iloc[fit_idx], y_dev.iloc[stop_idx]
                    fold_pre = make_preprocessor(development.iloc[fit_idx], scale_numeric=False)
                    X_fit_e = fold_pre.fit_transform(X_fit, y_fit)
                    X_stop_e = fold_pre.transform(X_stop)
                    model = CatBoostClassifier(**params)
                    model.fit(X_fit_e, y_fit, eval_set=[(X_stop_e, y_stop)],
                              use_best_model=True, early_stopping_rounds=35, verbose=False)
                    fold_scores.append(log_loss(y_stop, model.predict_proba(X_stop_e)[:, 1]))
                    trial.report(float(np.mean(fold_scores)), step=fold)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                return float(np.mean(fold_scores))

            study = optuna.create_study(
                direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
            )
            study.optimize(objective, n_trials=8 if FAST_MODE else 25, timeout=180 if FAST_MODE else 900)
            print("best CV log loss:", round(study.best_value, 4))
            display(pd.Series(study.best_params, name="best parameter"))
            """
        ),
        code(
            """
            trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "duration"))
            trials = trials.sort_values("value", ascending=False, na_position="last")
            display(trials.head(10))

            ax = study.trials_dataframe(attrs=("number", "value")).plot(
                x="number", y="value", marker="o", legend=False, figsize=(7, 3.5)
            )
            ax.set_title("Optuna optimization history")
            ax.set_xlabel("trial")
            ax.set_ylabel("development CV log loss")
            plt.show()
            """
        ),
        code(
            """
            try:
                importances = optuna.importance.get_param_importances(study)
                importance_frame = (
                    pd.Series(importances, name="importance")
                    .rename_axis("parameter")
                    .reset_index()
                    .sort_values("importance", ascending=True)
                )
                display(importance_frame.sort_values("importance", ascending=False))
                importance_frame.plot.barh(x="parameter", y="importance", legend=False, figsize=(7, 3.5))
                plt.title("Optuna parameter importance for this study")
                plt.xlabel("relative importance")
                plt.show()
            except Exception as exc:
                print(f"Parameter importance skipped: {exc}")
            """
        ),
        code(
            """
            tuned_params = {
                **study.best_params, "loss_function": "Logloss", "eval_metric": "Logloss",
                "thread_count": -1, "random_seed": SEED, "iterations": 1200,
                "allow_writing_files": False, "verbose": False, "bootstrap_type": "Bernoulli",
            }
            tuned = CatBoostClassifier(**tuned_params)
            tuned.fit(X_dev_encoded, y_dev, eval_set=[(X_val_encoded, y_val)],
                      use_best_model=True, early_stopping_rounds=50, verbose=False)
            tuned_p = tuned.predict_proba(X_val_encoded)[:, 1]
            validation_comparison = pd.DataFrame({
                "untuned CatBoost": classification_metrics(y_val, cat_p),
                "Optuna CatBoost": classification_metrics(y_val, tuned_p),
            }).T
            validation_comparison
            """
        ),
        md(
            """
            A tiny validation gain may not survive sampling variation and may not justify added search cost.
            Trial histories are adaptive; the best CV score is optimistically selected. Final claims wait for
            notebook 09's one-time test evaluation.

            ## Common mistakes and leakage warnings

            - Optimizing Optuna against validation repeatedly or against test even once.
            - Fitting one encoder before CV and reusing it across folds.
            - Letting early stopping inspect the same rows used to fit trees.
            - Searching huge spaces with too few trials, then overinterpreting the winner.
            - Treating parameter importance as universal truth instead of study-specific diagnostics.
            - Retuning the search space after looking at validation until the validation result improves.

            ## Exercises

            1. Plot Optuna parameter importance and explain why it is study-specific.
            2. Add a time budget and compare the best-so-far curve across two seeded studies.
            3. Add `scale_pos_weight` to the search space and explain how it changes threshold behavior.
            4. **Challenge:** implement nested CV for unbiased tuning-performance estimation and state its
               computational cost.

            ## Summary

            Optuna turns hyperparameter tuning into a reproducible study: define the objective, bound the
            search space, prune weak trials, inspect diagnostics, and promote one candidate. The objective
            stays inside the development boundary. Validation compares finalists, while test evaluation
            remains deferred.

            ## References

            - [Optuna first optimization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html)
            - [Optuna efficient optimization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)
            - [Optuna pruning](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#pruning-algorithms)
            - [CatBoostClassifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier)
            """
        ),
    ]


def n05() -> list[dict]:
    """Build notebook 05: ensemble learning."""
    return [
        md(
            """
            # 05 — Ensemble learning

            **Estimated time:** 110–150 minutes  
            **Prerequisites:** notebooks 00–04; bagging, boosting, and CV.  
            **Depends on:** leakage-safe preprocessing and development-only model selection.

            ## Learning objectives

            - Position Random Forest as a bagging baseline against boosting.
            - Measure model diversity through out-of-fold error correlation.
            - Build soft voting and leakage-safe stacking.
            - Compare performance, calibration, latency, and operational complexity.
            """
        ),
        COMMON,
        code(
            """
            import time
            import lightgbm as lgb
            from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold, cross_val_predict
            from sklearn.pipeline import Pipeline
            from src.course_utils import (classification_metrics, load_bank_data, make_preprocessor,
                                          make_splits, split_xy)

            development, validation, _sealed_test = make_splits(load_bank_data(), reduced=FAST_MODE)
            X_dev, y_dev = split_xy(development)
            X_val, y_val = split_xy(validation)
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

            warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier")

            def pipe(model, scale_numeric=False):
                return Pipeline([("preprocess", make_preprocessor(development, scale_numeric=scale_numeric)),
                                 ("model", model)])

            logistic = pipe(LogisticRegression(max_iter=1200, random_state=SEED), scale_numeric=True)
            forest = pipe(RandomForestClassifier(
                n_estimators=180 if FAST_MODE else 500, min_samples_leaf=4,
                max_features="sqrt", class_weight="balanced_subsample",
                n_jobs=-1, random_state=SEED,
            ))
            boosting = pipe(lgb.LGBMClassifier(
                n_estimators=240 if FAST_MODE else 500, learning_rate=.04,
                num_leaves=31, min_child_samples=40, subsample=.9,
                colsample_bytree=.9, random_state=SEED, n_jobs=-1, verbosity=-1,
            ))
            """
        ),
        md(
            """
            Random Forest averages decorrelated deep trees trained on bootstrap samples: variance falls, but
            probability estimates can be coarse and inference cost grows with tree count. Boosting fits trees
            sequentially to correct current errors: it often improves tabular accuracy but can overfit without
            shrinkage and structural regularization.

            Ensembling helps when models are individually competent and make different errors. Measuring
            diversity on in-sample predictions is optimistic, so we generate out-of-fold (OOF) probabilities.
            """
        ),
        code(
            """
            base = {"logistic": logistic, "random_forest": forest, "lightgbm": boosting}
            oof = {}
            for name, estimator in base.items():
                oof[name] = cross_val_predict(estimator, X_dev, y_dev, cv=cv,
                                              method="predict_proba", n_jobs=-1)[:, 1]
            oof_frame = pd.DataFrame(oof)
            display(oof_frame.corr())
            error_frame = oof_frame.sub(y_dev.to_numpy(), axis=0)
            display(error_frame.corr().style.format("{:.3f}"))
            """
        ),
        md(
            """
            ## Soft voting and stacking

            Soft voting averages probabilities and assumes they are on compatible scales; uncalibrated members
            can dominate. Stacking trains a meta-model on OOF base predictions. `StackingClassifier(cv=...)`
            generates those OOF predictions internally, then refits base learners on all development rows.
            Passing predictions fitted on the same rows would leak base-model overfit into the meta-model.
            """
        ),
        code(
            """
            voting = VotingClassifier(
                estimators=[("lr", logistic), ("rf", forest), ("lgb", boosting)],
                voting="soft", n_jobs=-1,
            )
            stacking = StackingClassifier(
                estimators=[("lr", logistic), ("rf", forest), ("lgb", boosting)],
                final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
                cv=cv, stack_method="predict_proba", passthrough=False, n_jobs=-1,
            )
            candidates = {**base, "soft_vote": voting, "stack": stacking}
            """
        ),
        code(
            """
            rows, fitted = [], {}
            for name, estimator in candidates.items():
                start = time.perf_counter()
                fitted[name] = estimator.fit(X_dev, y_dev)
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                p = fitted[name].predict_proba(X_val)[:, 1]
                infer_ms_per_1k = (time.perf_counter() - start) * 1000 / len(X_val) * 1000
                rows.append({"model": name, **classification_metrics(y_val, p),
                             "fit_seconds": fit_seconds, "inference_ms_per_1k": infer_ms_per_1k})
            ensemble_results = pd.DataFrame(rows).set_index("model")
            ensemble_results.sort_values("cost", ascending=True)
            """
        ),
        code(
            """
            from sklearn.calibration import CalibrationDisplay
            fig, ax = plt.subplots(figsize=(6, 5))
            for name in ["random_forest", "lightgbm", "soft_vote", "stack"]:
                CalibrationDisplay.from_predictions(
                    y_val, fitted[name].predict_proba(X_val)[:, 1], n_bins=8,
                    strategy="quantile", name=name, ax=ax,
                )
            ax.set_title("Validation calibration")
            plt.tight_layout()
            """
        ),
        md(
            """
            Stacking is not free: it multiplies training and inference paths, complicates monitoring and
            explanations, and can fail when base predictions shift differently. Prefer the simplest model
            inside an uncertainty-aware performance tolerance. If stacking's gain is tiny, operationally it
            is usually a loss.

            ## Common mistakes and leakage warnings

            - Training the meta-model on in-sample base predictions.
            - Comparing an ensemble to weak or under-tuned individual baselines.
            - Averaging badly calibrated probabilities without inspection.
            - Ignoring correlated failures, latency, artifact size, and dependency count.
            - Selecting the ensemble after repeated test comparisons.

            ## Exercises

            1. Optimize voting weights using development OOF predictions only.
            2. Calibrate each base learner, then reassess voting.
            3. **Challenge:** create a nested-CV stacking evaluation and compare its confidence interval with
               LightGBM's. Recommend a production model using an explicit complexity penalty.

            ## Summary

            Random Forest supplies a robust bagging reference; boosting often has stronger ranking. OOF
            predictions make diversity analysis and stacking honest. Ensembles earn deployment only when a
            reliable gain outweighs additional latency, explanation, monitoring, and maintenance cost.

            ## References

            - [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
            - [VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
            - [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
            """
        ),
    ]


def n06() -> list[dict]:
    """Build notebook 06: explainable AI with SHAP."""
    return [
        md(
            """
            # 06 — Explainable AI with SHAP

            **Estimated time:** 90–130 minutes  
            **Prerequisites:** notebooks 00–05; tree models and feature engineering.  
            **Depends on:** a validation-only LightGBM workflow.

            ## Learning objectives

            - Produce global and local SHAP explanations with the correct tree explainer.
            - Read beeswarm, bar, dependence, waterfall, and force-style plots.
            - Distinguish predictive attribution from causality.
            - Probe explanation stability and correlated-feature caveats.
            """
        ),
        COMMON,
        code(
            """
            import lightgbm as lgb
            import shap
            from scipy import sparse
            from src.course_utils import load_bank_data, make_preprocessor, make_splits, split_xy

            development, validation, _sealed_test = make_splits(load_bank_data(), reduced=FAST_MODE)
            X_dev, y_dev = split_xy(development)
            X_val, y_val = split_xy(validation)
            preprocessor = make_preprocessor(development, scale_numeric=False)
            X_dev_e = preprocessor.fit_transform(X_dev, y_dev)
            X_val_e = preprocessor.transform(X_val)
            feature_names = preprocessor.get_feature_names_out()
            model = lgb.LGBMClassifier(
                n_estimators=250 if FAST_MODE else 500, learning_rate=.04, num_leaves=31,
                min_child_samples=40, subsample=.9, colsample_bytree=.9,
                random_state=SEED, n_jobs=-1, verbosity=-1,
            ).fit(X_dev_e, y_dev)
            print("SHAP", shap.__version__, "features", len(feature_names))
            """
        ),
        md(
            """
            ## Explainer and background choices

            `TreeExplainer` uses tree structure for efficient exact or model-consistent attributions. We explain
            a reproducible validation sample, not training rows. One-hot feature names retain their transformer
            prefix so provenance is visible. SHAP decomposes a model output relative to a baseline; it does not
            recover the effect of an intervention.
            """
        ),
        code(
            """
            sample_n = 300 if FAST_MODE else 1000
            explain_rows = X_val.sample(n=min(sample_n, len(X_val)), random_state=SEED)
            X_explain = preprocessor.transform(explain_rows)
            if sparse.issparse(X_explain):
                X_explain = X_explain.toarray()
            X_explain = pd.DataFrame(X_explain, columns=feature_names, index=explain_rows.index)
            explainer = shap.TreeExplainer(model)
            explanation = explainer(X_explain)
            print("explanation shape:", explanation.shape)
            """
        ),
        md("## Global magnitude and direction"),
        code(
            """
            shap.plots.bar(explanation, max_display=15, show=False)
            plt.title("Global mean |SHAP| on validation sample")
            plt.tight_layout(); plt.show()
            shap.plots.beeswarm(explanation, max_display=15, show=False)
            plt.tight_layout(); plt.show()
            """
        ),
        md(
            """
            The bar plot ranks average attribution magnitude; it does not show sign or distribution. The
            beeswarm adds direction and heterogeneity. One-hot levels appear separately, so category-level
            magnitudes should sometimes be aggregated back to their source feature for communication.
            """
        ),
        code(
            """
            top_feature = feature_names[np.abs(explanation.values).mean(axis=0).argmax()]
            shap.plots.scatter(explanation[:, top_feature], color=explanation, show=False)
            plt.title(f"Dependence: {top_feature}")
            plt.tight_layout(); plt.show()
            """
        ),
        md("## Local explanation: one high-scoring client"),
        code(
            """
            probabilities = model.predict_proba(X_explain)[:, 1]
            local_position = int(np.argmax(probabilities))
            print("client row:", X_explain.index[local_position], "probability:", probabilities[local_position])
            shap.plots.waterfall(explanation[local_position], max_display=12, show=False)
            plt.tight_layout(); plt.show()
            shap.plots.force(explanation[local_position], matplotlib=True, show=False)
            plt.tight_layout(); plt.show()
            """
        ),
        md(
            """
            A local plot explains why this model output differs from its expected output. It does not say
            that changing a displayed feature will cause subscription. Correlated or substitutable predictors
            can split or reassign credit; one-hot indicators are structurally dependent.

            ## Stability probe

            We refit the same configuration with a different seed and compare global mean absolute SHAP.
            This is a lightweight diagnostic, not a complete stability study.
            """
        ),
        code(
            """
            model_2 = lgb.LGBMClassifier(
                n_estimators=250 if FAST_MODE else 500, learning_rate=.04, num_leaves=31,
                min_child_samples=40, subsample=.85, colsample_bytree=.85,
                random_state=SEED + 1, n_jobs=-1, verbosity=-1,
            ).fit(X_dev_e, y_dev)
            explanation_2 = shap.TreeExplainer(model_2)(X_explain)
            importance_1 = pd.Series(np.abs(explanation.values).mean(axis=0), index=feature_names)
            importance_2 = pd.Series(np.abs(explanation_2.values).mean(axis=0), index=feature_names)
            stability = importance_1.corr(importance_2, method="spearman")
            print("Spearman rank stability:", round(stability, 3))
            display(pd.DataFrame({"seed_42": importance_1, "seed_43": importance_2})
                    .sort_values("seed_42", ascending=False).head(15))
            """
        ),
        md(
            """
            Responsible communication reports the prediction moment, background/reference population,
            model version, feature definitions, uncertainty, known correlations, and a plain warning that
            attribution is not causation. Explanations can faithfully expose an undesirable model; they do
            not make that model fair or safe.

            ## Common mistakes and leakage warnings

            - Using a generic model-agnostic explainer when an efficient exact tree explainer is available.
            - Explaining training rows and presenting them as general behavior.
            - Reading SHAP direction as a causal intervention.
            - Ignoring preprocessing names and reporting transformed columns ambiguously.
            - Treating one explanation seed/sample as stable truth.

            ## Exercises

            1. Aggregate one-hot SHAP magnitudes to original features.
            2. Compare explanations for a false positive and false negative on validation.
            3. **Challenge:** bootstrap the validation sample 100 times and attach uncertainty intervals to
               the top-10 global importances; report rank instability.

            ## Summary

            SHAP provides model-relative global and local attributions. Correct explainer choice, explicit
            reference data, transformed-feature provenance, stability checks, and causal humility are part
            of the method—not optional communication polish.

            ## References

            - [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
            - [SHAP plots API](https://shap.readthedocs.io/en/latest/api.html#plots)
            - [Interpretable ML: SHAP chapter](https://christophm.github.io/interpretable-ml-book/shap.html)
            """
        ),
    ]


def n07() -> list[dict]:
    """Build notebook 07: optional anomaly detection extension."""
    return [
        md(
            """
            # 07 — Optional anomaly detection extension

            **Estimated time:** 75–100 minutes  
            **Prerequisites:** notebooks 00–03; distance, density, and ranking metrics.  
            **Depends on:** the same leakage-safe feature representation. **Optional module.**

            ## Learning objectives

            - Define an anomaly operationally rather than equating it with the rare target.
            - Compare Isolation Forest and novelty-mode Local Outlier Factor (LOF).
            - Evaluate ranked alerts with controlled contamination and precision@k.
            - Explain why anomalies, minority classes, noise, and fraud are different concepts.
            """
        ),
        COMMON,
        md(
            """
            ## A deliberately limited connection

            Bank Marketing has no verified anomaly or fraud label. Here an **anomaly** means a row that is
            unusual under the historical feature distribution and deserves data-quality or operations review.
            It does not mean subscriber, fraudster, or error. To obtain known evaluation labels, we inject a
            small set of transparent constraint-breaking records into a copy of validation data. This tests
            detection of those perturbations only; it does not validate real-world anomaly discovery.
            """
        ),
        code(
            """
            from sklearn.ensemble import IsolationForest
            from sklearn.neighbors import LocalOutlierFactor
            from sklearn.preprocessing import MaxAbsScaler
            from sklearn.pipeline import Pipeline
            from src.course_utils import load_bank_data, make_preprocessor, make_splits, split_xy

            development, validation, _sealed_test = make_splits(load_bank_data(), reduced=FAST_MODE)
            X_dev, _ = split_xy(development)
            X_val, _ = split_xy(validation)
            preprocessor = make_preprocessor(development, scale_numeric=True)
            Z_dev = preprocessor.fit_transform(X_dev)
            """
        ),
        code(
            """
            rng = np.random.default_rng(SEED)
            contaminated = X_val.copy().reset_index(drop=True)
            anomaly_label = np.zeros(len(contaminated), dtype=int)
            injected_idx = rng.choice(len(contaminated), size=max(40, len(contaminated) // 50), replace=False)
            anomaly_label[injected_idx] = 1
            # Controlled data-integrity/operational violations, chosen to be explicit and reproducible.
            third = len(injected_idx) // 3
            contaminated.loc[injected_idx[:third], "age"] = 120
            contaminated.loc[injected_idx[third:2*third], "campaign"] = 250
            contaminated.loc[injected_idx[2*third:], "balance"] = 1_000_000
            Z_eval = preprocessor.transform(contaminated)
            print("injected prevalence:", anomaly_label.mean())
            """
        ),
        md(
            """
            Isolation Forest isolates observations using random partitions; fewer splits imply greater
            abnormality. LOF compares local density with neighbors. `novelty=True` is essential when scoring
            new rows: default LOF is intended for outlier detection on its training set and exposes a different
            prediction interface. Both are sensitive to representation and hyperparameters.
            """
        ),
        code(
            """
            detectors = {
                "IsolationForest": IsolationForest(
                    n_estimators=200 if FAST_MODE else 500, contamination="auto",
                    random_state=SEED, n_jobs=-1,
                ),
                "LOF_novelty": LocalOutlierFactor(n_neighbors=35, novelty=True, contamination="auto", n_jobs=-1),
            }
            scores = {}
            for name, detector in detectors.items():
                detector.fit(Z_dev)
                scores[name] = -detector.decision_function(Z_eval)  # larger = more anomalous
            score_frame = pd.DataFrame(scores)
            """
        ),
        code(
            """
            def precision_at_k(labels, score, k):
                order = np.argsort(score)[::-1][:k]
                return float(np.mean(np.asarray(labels)[order]))

            def recall_at_k(labels, score, k):
                order = np.argsort(score)[::-1][:k]
                positives = np.asarray(labels).sum()
                return float(np.asarray(labels)[order].sum() / positives) if positives else np.nan

            k = int(anomaly_label.sum())
            evaluation = pd.DataFrame({name: {
                "precision_at_k": precision_at_k(anomaly_label, score, k),
                "recall_at_k": recall_at_k(anomaly_label, score, k),
                "lift_at_k": precision_at_k(anomaly_label, score, k) / anomaly_label.mean(),
                "k": k,
            } for name, score in scores.items()}).T
            display(evaluation)

            plot_data = score_frame.assign(injected=np.where(anomaly_label == 1, "injected", "original"))
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for ax, name in zip(axes, scores):
                sns.histplot(data=plot_data, x=name, hue="injected", bins=40,
                             stat="density", common_norm=False, element="step", ax=ax)
            plt.tight_layout()
            """
        ),
        md(
            """
            Precision@k matches a fixed investigation budget, while recall@k shows how many injected anomalies
            the review budget captured. Original rows scoring highly may be valid
            rare customers, genuine data errors, distribution tails, or artifacts of encoding. They require
            review—not automatic deletion.

            ## Common mistakes and leakage warnings

            - Calling the positive supervised class an anomaly merely because it is rare.
            - Evaluating on the same rows used to fit an outlier detector.
            - Using LOF's training-outlier mode to score future observations.
            - Deleting anomalies before understanding whether they are valid business cases.
            - Claiming fraud detection without a fraud definition or outcome labels.

            ## Exercises

            1. Inject subtler multivariate anomalies that preserve all univariate ranges.
            2. Compare precision@k across neighborhood sizes and random seeds.
            3. **Challenge:** design a human-review evaluation for naturally high-scoring rows, including a
               sampling plan that avoids verification bias.

            ## Summary

            Anomaly detection is optional because this dataset supplies no natural anomaly label. Controlled
            contamination makes the evaluation honest about what it measures. Ranked alerts can support review,
            but anomaly score, rare class, noise, and fraud are not interchangeable.

            ## References

            - [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
            - [LocalOutlierFactor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
            - [Outlier and novelty detection guide](https://scikit-learn.org/stable/modules/outlier_detection.html)
            """
        ),
    ]


def n08() -> list[dict]:
    """Build notebook 08: data drift and model monitoring."""
    return [
        md(
            """
            # 08 — Data drift and model monitoring

            **Estimated time:** 110–150 minutes  
            **Prerequisites:** notebooks 00–06; hypothesis tests, calibration, and divergence.  
            **Depends on:** a fixed preprocessing/model artifact and delayed-label assumptions.

            ## Learning objectives

            - Distinguish covariate, label/prevalence, concept, prediction, and performance drift.
            - Compute PSI, KS, chi-square, and Jensen–Shannon diagnostics with caveats.
            - Simulate plausible campaign drift and monitor predictions/calibration.
            - Define alerts, delayed-label workflows, investigation, and retraining gates.
            """
        ),
        COMMON,
        code(
            """
            import lightgbm as lgb
            from scipy.spatial.distance import jensenshannon
            from scipy.stats import ks_2samp, chi2_contingency
            from sklearn.pipeline import Pipeline
            from src.course_utils import (classification_metrics, load_bank_data, make_preprocessor,
                                          make_splits, split_xy)

            development, validation, _sealed_test = make_splits(load_bank_data(), reduced=FAST_MODE)
            X_ref, y_ref = split_xy(development)
            X_current, y_current = split_xy(validation)
            model = Pipeline([
                ("preprocess", make_preprocessor(development, scale_numeric=False)),
                ("model", lgb.LGBMClassifier(n_estimators=250, learning_rate=.04, num_leaves=31,
                                             min_child_samples=40, random_state=SEED, n_jobs=-1, verbosity=-1)),
            ]).fit(X_ref, y_ref)
            """
        ),
        md(
            """
            ## Drift vocabulary

            - **Covariate drift:** $P(X)$ changes.
            - **Label/prevalence drift:** $P(Y)$ changes.
            - **Concept drift:** the conditional relationship $P(Y | X)$ changes; feature checks alone cannot establish it.
            - **Prediction drift:** the score distribution changes, possibly due to input drift or model changes.
            - **Performance degradation:** labeled metrics worsen; this is the outcome monitoring ultimately cares about.

            We simulate an older, more heavily contacted current population and a channel-mix shift. Separately,
            we create a prevalence-shifted sample. These are teaching scenarios, not claims about actual bank history.
            """
        ),
        code(
            """
            rng = np.random.default_rng(SEED)
            X_drift = X_current.copy()
            X_drift["age"] = (X_drift["age"] + rng.integers(3, 10, len(X_drift))).clip(18, 95)
            X_drift["campaign"] = X_drift["campaign"] + rng.poisson(2, len(X_drift))
            channel_change = rng.random(len(X_drift)) < .30
            X_drift.loc[channel_change, "contact"] = "telephone"

            positive = validation[validation[TARGET] == 1]
            negative = validation[validation[TARGET] == 0]
            prevalence_drift = pd.concat([
                positive.sample(n=min(len(positive), 650), replace=True, random_state=SEED),
                negative.sample(n=2350, replace=True, random_state=SEED),
            ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
            print("reference/current/prevalence-shift rates:", y_ref.mean(), y_current.mean(), prevalence_drift[TARGET].mean())
            """
        ),
        md(
            """
            ## Numeric checks: PSI, KS, and Jensen–Shannon

            PSI bins the reference, then sums `(q_i - p_i) × log(q_i / p_i)`. It is asymmetric, bin-sensitive,
            lacks a universal sampling distribution, and conventional 0.1/0.25 cutoffs are heuristics.
            KS tests equality of continuous distributions but becomes hypersensitive with large samples.
            Jensen–Shannon divergence is symmetric and bounded after choosing a histogram discretization.
            """
        ),
        code(
            """
            def numeric_drift(reference, current, bins=10):
                edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
                if len(edges) < 3:
                    return {"psi": 0.0, "ks_stat": 0.0, "ks_p": 1.0, "js": 0.0}
                edges[0], edges[-1] = -np.inf, np.inf
                p = np.histogram(reference, bins=edges)[0].astype(float)
                q = np.histogram(current, bins=edges)[0].astype(float)
                p = np.clip(p / p.sum(), 1e-6, None); q = np.clip(q / q.sum(), 1e-6, None)
                ks = ks_2samp(reference, current)
                return {"psi": float(np.sum((q-p) * np.log(q/p))),
                        "ks_stat": ks.statistic, "ks_p": ks.pvalue,
                        "js": float(jensenshannon(p, q, base=2) ** 2)}

            numeric_columns = X_ref.select_dtypes(include=np.number).columns
            numeric_report = pd.DataFrame({c: numeric_drift(X_ref[c], X_drift[c]) for c in numeric_columns}).T
            numeric_report.sort_values("psi", ascending=False)
            """
        ),
        md("## Categorical checks and multiple testing"),
        code(
            """
            def categorical_drift(reference, current):
                levels = sorted(set(reference.astype(str)) | set(current.astype(str)))
                a = reference.astype(str).value_counts().reindex(levels, fill_value=0)
                b = current.astype(str).value_counts().reindex(levels, fill_value=0)
                table = np.vstack([a, b])
                chi2, p, _, _ = chi2_contingency(table)
                pa = np.clip(a / a.sum(), 1e-6, None); pb = np.clip(b / b.sum(), 1e-6, None)
                return {"chi2": chi2, "p_value": p,
                        "js": float(jensenshannon(pa, pb, base=2) ** 2)}

            categorical_columns = X_ref.select_dtypes(include="object").columns
            categorical_report = pd.DataFrame({
                c: categorical_drift(X_ref[c], X_drift[c]) for c in categorical_columns
            }).T

            # Benjamini-Hochberg adjusted p-values control false discovery rate across this family of checks.
            def bh_adjust(p_values):
                p = np.asarray(p_values); order = np.argsort(p); ranked = p[order]
                adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
                out = np.empty_like(adjusted); out[order] = np.clip(adjusted, 0, 1)
                return out
            categorical_report["p_value_bh"] = bh_adjust(categorical_report["p_value"])
            categorical_report.sort_values("js", ascending=False)
            """
        ),
        md(
            """
            Statistical significance is not operational importance. Sample size, effect magnitude, persistence,
            feature criticality, and downstream performance all matter. Tests also assume independent samples;
            repeated customers or campaign clustering would violate that assumption.

            ## Prediction now, performance later
            """
        ),
        code(
            """
            p_ref = model.predict_proba(X_ref)[:, 1]
            p_current = model.predict_proba(X_current)[:, 1]
            p_drift = model.predict_proba(X_drift)[:, 1]
            prediction_report = pd.DataFrame({
                "reference": numeric_drift(p_ref, p_ref),
                "natural_validation": numeric_drift(p_ref, p_current),
                "simulated_covariate_drift": numeric_drift(p_ref, p_drift),
            }).T
            display(prediction_report)
            display(pd.DataFrame({
                "development (optimistic in-sample)": classification_metrics(y_ref, p_ref),
                "validation with arrived labels": classification_metrics(y_current, p_current),
            }).T)
            """
        ),
        code(
            """
            from sklearn.calibration import CalibrationDisplay
            CalibrationDisplay.from_predictions(y_current, p_current, n_bins=8, strategy="quantile")
            plt.title("Calibration once validation labels arrive")
            plt.tight_layout(); plt.show()

            dashboard = pd.DataFrame([
                {"signal": "age PSI", "value": numeric_report.loc["age", "psi"], "warn": 0.10, "critical": 0.25},
                {"signal": "campaign PSI", "value": numeric_report.loc["campaign", "psi"], "warn": 0.10, "critical": 0.25},
                {"signal": "prediction PSI", "value": prediction_report.loc["simulated_covariate_drift", "psi"], "warn": 0.10, "critical": 0.25},
            ])
            dashboard["status"] = np.select(
                [dashboard.value >= dashboard.critical, dashboard.value >= dashboard.warn],
                ["critical", "warning"], default="ok")
            dashboard
            """
        ),
        md(
            """
            With delayed labels, monitor schema failures, missing/unknown rates, feature and prediction drift,
            volumes, latency, and decision rates immediately. Join outcomes later by prediction ID to update
            cost, precision, recall, subgroup behavior, and calibration. Never retrain solely because one p-value fires.

            **Investigation workflow:** validate pipeline/data changes → localize segments/features → check label
            quality and delay → measure business/performance impact → reproduce offline → approve retraining or
            rollback → shadow/canary → document. A retraining trigger should combine sustained material drift,
            labeled degradation beyond tolerance, and a candidate that passes the same validation gates.

            ## Common mistakes and leakage warnings

            - Calling covariate drift concept drift without labels.
            - Using universal PSI thresholds without baseline variability.
            - Alerting on dozens of unadjusted p-values.
            - Comparing raw category codes with numeric distance tests.
            - Retraining automatically on corrupted or incompletely labeled data.

            ## Exercises

            1. Build bootstrap control limits from development windows instead of fixed PSI heuristics.
            2. Simulate a conditional label flip to represent concept drift and show why $P(X)$ checks miss it.
            3. **Challenge:** design a delayed-label monitor with prediction IDs, event-time windows, backfill,
               and alert deduplication.

            ## Summary

            Drift statistics are diagnostic signals, not deployment decisions. Monitoring joins immediate data
            and prediction signals with later performance/calibration evidence, then routes sustained material
            changes through investigation and governed retraining.

            ## References

            - [SciPy KS test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
            - [SciPy chi-square contingency test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)
            - [SciPy Jensen–Shannon distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html)
            - [NIST multiple comparisons overview](https://www.itl.nist.gov/div898/handbook/prc/section4/prc47.htm)
            """
        ),
    ]


def n09() -> list[dict]:
    """Build notebook 09: end-to-end production ML project."""
    return [
        md(
            """
            # 09 — End-to-end production ML project

            **Estimated time:** 150–210 minutes  
            **Prerequisites:** notebooks 00–08.  
            **Depends on:** every prior data, evaluation, leakage, explanation, and monitoring decision.

            ## Learning objectives

            - Rebuild a clean train-to-inference pipeline with a data contract.
            - Select a threshold on validation, then evaluate the sealed test exactly once.
            - Serialize, reload, and verify deterministic batch inference.
            - Produce reproducibility metadata, checks, a model card, and monitoring plan.
            """
        ),
        COMMON,
        md(
            """
            ## Final protocol

            We choose a regularized LightGBM pipeline as a pragmatic performance/complexity compromise. The
            estimator and feature logic are fixed before test access. Development fits the model; validation
            selects the cost-sensitive threshold; the unchanged model and threshold are then evaluated once on
            test. We do not refit on development+validation before this evaluation because that would change the
            calibrated operating behavior selected on validation.
            """
        ),
        code(
            """
            import json, joblib, lightgbm as lgb
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import FunctionTransformer
            from src.course_utils import (SEED, TARGET, add_domain_features, classification_metrics,
                                          environment_metadata, file_sha256, load_bank_data, make_preprocessor,
                                          make_splits, project_root, split_xy, threshold_table, write_json)

            full = load_bank_data()
            development, validation, test = make_splits(full, reduced=FAST_MODE)
            X_dev, y_dev = split_xy(development)
            X_val, y_val = split_xy(validation)
            X_test, y_test = split_xy(test)  # first notebook that permits test labels

            engineered_schema = add_domain_features(development)
            final_pipeline = Pipeline([
                ("features", FunctionTransformer(add_domain_features, validate=False)),
                ("preprocess", make_preprocessor(engineered_schema, scale_numeric=False)),
                ("model", lgb.LGBMClassifier(
                    n_estimators=280 if FAST_MODE else 500, learning_rate=.04, num_leaves=31,
                    min_child_samples=40, subsample=.9, colsample_bytree=.9,
                    reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1,
                )),
            ])
            final_pipeline.fit(X_dev, y_dev)
            """
        ),
        md("## Validation threshold selection; then one-time final test evaluation"),
        code(
            """
            validation_probability = final_pipeline.predict_proba(X_val)[:, 1]
            threshold_result = threshold_table(y_val, validation_probability).sort_values("cost").iloc[0]
            selected_threshold = float(threshold_result["threshold"])
            print("frozen threshold:", selected_threshold)
            display(pd.DataFrame({
                "validation": classification_metrics(y_val, validation_probability, selected_threshold)
            }).T)

            # FINAL TEST ACCESS: do not use these results to revise features, hyperparameters, or threshold.
            test_probability = final_pipeline.predict_proba(X_test)[:, 1]
            final_test_metrics = classification_metrics(y_test, test_probability, selected_threshold)
            display(pd.DataFrame({"final_test": final_test_metrics}).T)
            """
        ),
        md(
            """
            The final test estimate has sampling uncertainty and applies to this historical distribution. If
            the result disappoints, it remains the honest result for this development cycle. Any subsequent
            redesign starts a new cycle and needs fresh final-evaluation data.

            ## Data contract and batch inference
            """
        ),
        code(
            """
            REQUIRED_COLUMNS = X_dev.columns.tolist()
            NUMERIC_COLUMNS = X_dev.select_dtypes(include=np.number).columns.tolist()
            CATEGORICAL_COLUMNS = X_dev.select_dtypes(include="object").columns.tolist()

            def validate_batch(frame):
                if not isinstance(frame, pd.DataFrame):
                    raise TypeError("batch must be a pandas DataFrame")
                missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
                extra = sorted(set(frame.columns) - set(REQUIRED_COLUMNS))
                if missing or extra:
                    raise ValueError(f"schema mismatch; missing={missing}, extra={extra}")
                if frame.empty:
                    raise ValueError("batch must contain at least one row")
                numeric = frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
                if numeric.isna().any().any():
                    bad = numeric.columns[numeric.isna().any()].tolist()
                    raise ValueError(f"non-numeric or missing values in required numeric columns: {bad}")
                if not frame["age"].between(18, 120).all():
                    raise ValueError("age outside supported [18, 120] range")
                return frame[REQUIRED_COLUMNS].copy()

            def predict_batch(frame, pipeline=final_pipeline, threshold=selected_threshold):
                valid = validate_batch(frame)
                probability = pipeline.predict_proba(valid)[:, 1]
                return pd.DataFrame({
                    "subscription_probability": probability,
                    "prioritize_call": probability >= threshold,
                    "model_threshold": threshold,
                }, index=valid.index)

            batch_result = predict_batch(X_val.head(5))
            batch_result
            """
        ),
        md("## Serialization and reload verification"),
        code(
            """
            artifact_path = project_root() / "models" / "bank_marketing_pipeline.joblib"
            metadata_path = project_root() / "reports" / "reproducibility_metadata.json"
            joblib.dump({"pipeline": final_pipeline, "threshold": selected_threshold,
                         "required_columns": REQUIRED_COLUMNS}, artifact_path)
            reloaded = joblib.load(artifact_path)
            before = final_pipeline.predict_proba(X_val.head(50))[:, 1]
            after = reloaded["pipeline"].predict_proba(X_val.head(50))[:, 1]
            np.testing.assert_allclose(before, after, rtol=0, atol=1e-12)

            source_path = project_root() / "data" / "raw" / "bank-full.csv"
            metadata = {
                **environment_metadata(), "seed": SEED, "fast_mode": FAST_MODE,
                "data_sha256": file_sha256(source_path), "threshold": selected_threshold,
                "development_rows": len(development), "validation_rows": len(validation),
                "test_rows": len(test), "final_test_metrics": final_test_metrics,
            }
            write_json(metadata, metadata_path)
            print("reload predictions match; artifacts:", artifact_path, metadata_path)
            """
        ),
        md("## Basic unit-style checks and failure handling"),
        code(
            """
            assert "duration" not in REQUIRED_COLUMNS
            assert set(predict_batch(X_val.head(3)).columns) == {
                "subscription_probability", "prioritize_call", "model_threshold"}
            assert predict_batch(X_val.head(3))["subscription_probability"].between(0, 1).all()
            try:
                predict_batch(X_val.head(1).drop(columns="age"))
                raise AssertionError("missing-column check did not fire")
            except ValueError as exc:
                assert "missing=['age']" in str(exc)
            try:
                invalid = X_val.head(1).copy(); invalid["age"] = 999
                predict_batch(invalid)
                raise AssertionError("range check did not fire")
            except ValueError as exc:
                assert "age outside" in str(exc)
            print("all unit-style checks passed")
            """
        ),
        md(
            """
            ## Model card

            **Purpose:** prioritize clients immediately before a scheduled term-deposit marketing call.  
            **Users:** authorized campaign analysts; not an autonomous eligibility or credit system.  
            **Training data:** UCI Bank Marketing, Portuguese institution, historical campaigns; CC BY 4.0.  
            **Target:** observed subscription after contact. This is prediction, not treatment effect.  
            **Inputs:** 15 pre-contact fields; current-call `duration` is prohibited.  
            **Model:** engineered mixed-type pipeline with one-hot encoding and LightGBM.  
            **Threshold:** chosen on validation under illustrative cost `FP + 5 × FN`.  
            **Evaluation:** test accessed once above; inspect the stored metrics in this executed notebook.  
            **Limitations:** no customer ID or complete timestamp; possible repeated-client leakage; historical
            and geographic transport risk; unknown fairness impact; probabilities may drift.  
            **Out of scope:** causal targeting, credit decisions, fraud detection, or use outside governed campaigns.

            ## Monitoring plan

            Log model/data versions, prediction ID, event time, schema outcomes, score, decision, and latency.
            Immediately monitor volume, schema failures, unknown/missing rates, feature/prediction drift, and
            decision rate. When labels arrive, join by prediction ID and monitor precision, recall, business cost,
            calibration, and approved subgroup slices. Investigate sustained alerts before retraining; require
            reproducible data, challenger validation, governance review, rollback, and shadow/canary checks.

            ## Deployment architecture (cloud-neutral)

            A versioned training job emits the pipeline, threshold, metadata, and model card to an artifact
            registry. A scheduled batch scorer validates an immutable input snapshot, writes versioned
            predictions, and emits monitoring events. A later outcome job joins labels. Orchestration, secret
            management, access control, audit logs, and rollback surround these components; a notebook is not
            itself a production service.

            ## Common mistakes and leakage warnings

            - Refitting on validation before test while retaining a threshold selected for the old model.
            - Serializing only the estimator and reimplementing preprocessing in serving code.
            - Accepting extra/missing columns silently.
            - Treating a successful reload as proof of cross-version portability.
            - Retraining on drift alerts without label-quality and performance investigation.

            ## Exercises

            1. Add allowed-category checks with a policy for genuinely new categories.
            2. Write a batch contract containing prediction IDs and event timestamps.
            3. **Challenge:** package the scorer behind a local API with request validation, structured logs,
               concurrency limits, health checks, and a rollback test—without changing model semantics.

            ## Summary

            The course closes with one artifact spanning deterministic data access, leakage-safe fitting,
            validation-only threshold selection, one-time test evaluation, serialization, guarded inference,
            reproducibility evidence, model documentation, and a delayed-label monitoring plan.

            ## References

            - [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html)
            - [scikit-learn pipelines](https://scikit-learn.org/stable/modules/compose.html#pipeline)
            - [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
            - [Google model cards paper](https://doi.org/10.1145/3287560.3287596)
            """
        ),
    ]


BUILDERS = {
    "00_course_setup_and_dataset.ipynb": n00,
    "01_gradient_boosting_fundamentals.ipynb": n01_gradient_boosting,
    "02_advanced_feature_engineering.ipynb": n02,
    "03_imbalanced_learning.ipynb": n03,
    "04_optuna_hyperparameter_optimization.ipynb": n04,
    "05_ensemble_learning.ipynb": n05,
    "06_anomaly_detection_extension.ipynb": n06,
    "07_end_to_end_production_ml_project.ipynb": n07,
}


def main(names: list[str]) -> None:
    """Generate the requested course notebooks."""
    NB_DIR.mkdir(parents=True, exist_ok=True)
    selected = names or list(BUILDERS)
    for name in selected:
        if name not in BUILDERS:
            raise SystemExit(f"Unknown notebook: {name}")
        path = NB_DIR / name
        path.write_text(json.dumps(notebook(BUILDERS[name]()), indent=1), encoding="utf-8")
        print("built", path.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
