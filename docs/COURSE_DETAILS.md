# Advanced Production-Oriented Machine Learning

## Course details for modules 00–02

This document describes the first three modules of the course and their local video lessons. Video durations are rounded to the nearest second.

## Course overview

The course follows one production-oriented tabular classification project from problem framing to model development. These first three modules establish the foundation:

1. **Module 00 — Course setup and dataset:** define the business question, prediction moment, legal data, validation contracts, and a trustworthy baseline.
2. **Module 01 — Gradient boosting fundamentals:** understand how boosted trees learn, how their scores become probabilities and decisions, and how capacity is controlled.
3. **Module 02 — Advanced feature engineering:** turn domain hypotheses into safe features and understand how CatBoost handles categories, missing values, interactions, and model governance.

The running dataset is the UCI Bank Marketing dataset. The course treats the task as **prioritizing customers by predicted subscription probability**, not estimating whether a phone call causes a subscription. The production prediction moment is immediately before a scheduled call; therefore, the current call’s `duration` is unavailable and is treated as target leakage.

## Audience and prerequisites

The material assumes that learners already know:

- Python and common data-science tools such as pandas and scikit-learn;
- basic supervised learning and classification terminology;
- introductory statistics and model evaluation;
- the general purpose of train, validation, and test splits.

The modules add the production and senior-practitioner perspective: every feature needs a meaning, an owner, a valid time of availability, and an evaluation protocol that can be trusted.

## At a glance

| Module | Concept notes | Videos | Video time | Main outcome |
|---|---:|---:|---:|---|
| 00 — Course setup and dataset | 11 | 12 | 2:52:08 | A leakage-aware prediction and evaluation contract |
| 01 — Gradient boosting fundamentals | 6 | 7 | 1:15:59 | A clear mental model for boosted-tree learning and capacity |
| 02 — Advanced feature engineering | 12 | 11 | 1:43:37 | Safe, hypothesis-driven features and CatBoost design knowledge |
| **Total** | **29** | **30** | **5:51:45** | A foundation for reliable model development |

## Module 00 — Course setup and dataset

### Purpose

This module establishes what the model is allowed to predict, which data it may use, and how evidence will be collected. It starts with the distinction between prediction and causal effect, then moves through data governance, feature availability, leakage, preprocessing, baseline modeling, and evaluation design.

### Concepts covered

| # | Concept | Core idea |
|---:|---|---|
| 00 | Prediction vs. causal effect | A predictive model estimates the probability of an outcome given observed features and ranks likely outcomes. It does not automatically estimate the incremental effect of an intervention such as a phone call. |
| 01 | Validation contracts for production ML | Validate both the incoming dataset and each prediction pipeline request. Schema, type, range, category, and missing-value failures should be visible rather than silently absorbed. |
| 02 | Data governance and feature contracts | A data dictionary describes a field; a feature contract also records availability, ownership, refresh behavior, safe-use conditions, and validation rules. |
| 03 | Two-layer data-contract validation | Ingestion validation protects the data platform, while pipeline validation protects transformations and model serving. Both layers are needed because corruption can occur at every boundary. |
| 04 | Prediction timeline and leakage | Leakage is information that would not exist when the model is asked to predict. The correct question is temporal: could the feature be supplied at the prediction moment? |
| 05 | Feature availability and prediction moment | A feature is valid only in the context of a defined decision point. The same column can be legal for one scoring workflow and illegal for another. |
| 06 | Preprocessing inside cross-validation | Learned preprocessing such as imputation, scaling, encoding, and feature selection must be fitted inside each training fold to avoid train–validation contamination. |
| 07 | Logistic regression as a production baseline | A simple baseline provides a reference for whether added model complexity is justified. The decision should consider operational cost, interpretability, and deployment fit, not only AUC. |
| 08 | The test set as future evaluation | Keep the test set sealed until the workflow is frozen. Development and validation data support fitting and selection; the test set provides one final estimate of future performance. |
| 09 | Temporal and entity-aware validation | Random stratification preserves class proportions but may still leak time or entity information. When the domain requires it, validation must respect chronology, groups, or both. |
| 10 | Probability quality vs. decision quality | Probability estimation, ranking, calibration, thresholding, and business action are different layers. A good classifier score does not by itself define the right decision policy. |

### Video lessons

| Lesson | Video file | Duration |
|---|---|---:|
| Coding walkthrough: course setup and dataset | `00_code_course_setup_and_dataset.mp4` | 26:46 |
| Prediction vs. causal effect | `00_prediction_vs_causal_effect.mp4` | 26:36 |
| Validation contracts for production ML | `01_validation_contracts_for_production_ml.mp4` | 23:59 |
| Data governance and feature contracts | `02_data_governance_and_feature_contracts.mp4` | 16:35 |
| Data contracts and two-layer validation | `03_data_contracts_two_layer_validation.mp4` | 9:02 |
| Prediction timeline and leakage | `04_prediction_timeline_and_leakage.mp4` | 11:30 |
| Feature availability and prediction moment | `05_feature_availability_and_prediction_moment.mp4` | 7:07 |
| Preprocessing inside cross-validation | `06_preprocessing_inside_cross_validation.mp4` | 11:45 |
| Logistic regression as a production baseline | `07_logistic_regression_as_production_baseline.mp4` | 10:59 |
| Test set as future evaluation | `08_test_set_as_future_evaluation.mp4` | 9:32 |
| Temporal and entity-aware validation | `09_temporal_and_entity_aware_validation.mp4` | 9:59 |
| Probability quality vs. decision quality | `10_probability_quality_vs_decision_quality.mp4` | 8:18 |
| **Module 00 total** |  | **2:52:08** |

### Module outcome

By the end of Module 00, the learner should be able to explain the prediction moment, identify why `duration` is illegal for pre-call scoring, define ingestion and pipeline checks, fit preprocessing without leakage, compare against a logistic-regression baseline, and reserve the test set for final evaluation.

## Module 01 — Gradient boosting fundamentals

### Purpose

This module explains gradient boosting from the inside out. The learner moves from sequential additive trees to pseudo-residuals and raw scores, then separates probability estimation from decision thresholds. The final concepts connect tree depth and boosting parameters to interaction order, capacity, regularization, and model selection.

### Concepts covered

| # | Concept | Core idea |
|---:|---|---|
| 00 | Sequential additive boosting | A boosted model starts with a simple prediction and adds weak learners sequentially, with each new tree correcting the current model’s remaining error. |
| 01 | Pseudo-residuals and raw scores | For classification, the correction signal is derived from the loss gradient rather than ordinary residuals. The model learns an additive raw score that is later transformed into a probability. |
| 02 | Model, probability, and decision layers | Raw model scores, predicted probabilities, and business decisions are separate layers. Changing a threshold changes actions without changing the underlying score model. |
| 03 | Tree depth as interaction order | Tree depth controls the complexity of interactions a weak learner can express. Shallow trees capture simpler combinations; deeper trees can represent higher-order rules but overfit more easily. |
| 04 | Boosting capacity and regularization | `n_estimators`, `learning_rate`, `max_depth`, subsampling, minimum leaf size, regularization, and early stopping jointly determine model capacity. These parameters should be reasoned about as an interacting system. |
| 05 | Validation as model selection | Repeatedly comparing models on one validation set turns that set into part of the selection process. Early stopping is also model selection because it chooses the best iteration from many candidate models. |

### Video lessons

| Lesson | Video file | Duration |
|---|---|---:|
| Sequential additive boosting | `00_sequential_additive_boosting.mp4` | 21:12 |
| Coding walkthrough: gradient boosting fundamentals | `01_code_gradient_boosting_fundamentals.mp4` | 18:01 |
| Pseudo-residuals and raw scores | `01_pseudo_residuals_and_raw_scores.mp4` | 5:45 |
| Model, probability, and decision layers | `02_model_probability_and_decision_layers.mp4` | 5:14 |
| Tree depth as interaction order | `03_tree_depth_as_interaction_order.mp4` | 7:28 |
| Boosting capacity and regularization | `04_boosting_capacity_and_regularization.mp4` | 11:01 |
| Validation as model selection | `05_validation_as_model_selection.mp4` | 7:18 |
| **Module 01 total** |  | **1:15:59** |

### Module outcome

By the end of Module 01, the learner should be able to describe the additive training loop, distinguish raw scores from probabilities and decisions, explain how tree depth represents interaction complexity, reason about the main capacity controls, and recognize validation reuse as a source of optimistic model selection.

## Module 02 — Advanced feature engineering

### Purpose

This module treats feature engineering as a governed modeling activity rather than a collection of clever transformations. It begins with prediction-time contracts and business hypotheses, then covers ablation studies, safe categorical encoding, CatBoost’s ordered procedures, automatic interactions, native missing-value handling, symmetric trees, early stopping, model constraints, inspection, text, and embeddings.

### Concepts covered

| # | Concept | Core idea | Matching video |
|---:|---|---|---|
| 00 | Prediction-time data contracts and semantic features | A feature must be available at scoring time, represent the intended business meaning, and be produced through a leakage-safe transformation. Special values such as `pdays = -1` need semantic treatment rather than automatic conversion to missing. | `00_prediction_time_data_contract_and_semantic_features.mp4` |
| 01 | Feature engineering as a business hypothesis | A new feature is a hypothesis about behavior. Define why it should help, when it can fail, whether it is available at prediction time, and how its value will be tested. | `01_feature_engineering_as_business_hypothesis.mp4` |
| 02 | Ablation studies over feature storytelling | Compare feature families against a baseline and remove them systematically. A compelling explanation is not evidence of value; improvement must be measured for size, stability, and engineering cost. | `02_ablation_studies_over_feature_storytelling.mp4` |
| 03 | Production-safe categorical encoding | One-hot encoding can be safe when fitted correctly and configured for unknown categories, but high-cardinality features create memory, sparsity, and coefficient-stability concerns. | `03_production_safe_categorical_encoding.mp4` |
| 04 | CatBoost ordered category statistics | Naive target encoding can leak the target, especially for rare categories. CatBoost uses ordered statistics with a prior and a permutation-based construction to reduce this leakage and overfitting. | `04_catboost_ordered_category_statistics.mp4` |
| 05 | Ordered boosting and prediction shift | Standard boosting can use information that makes training-time residuals unlike prediction-time residuals. Ordered boosting reduces this prediction shift by constructing corrections with more appropriate historical information. | `05_ordered_boosting_and_prediction_shift.mp4` |
| 06 | Tree ensembles and automatic interactions | Trees can discover interactions without manually creating every cross-feature. Feature engineering remains important for semantics, availability, and representation, but it should not duplicate patterns the model can learn reliably. | `06_tree_ensembles_automatic_interactions.mp4` |
| 07 | CatBoost native missing-value splits | Missingness can be predictive, so CatBoost can learn missing-value behavior directly. This is different from treating a legitimate business sentinel such as `-1` as an ordinary missing value. | `07_catboost_native_missing_value_splits.mp4` |
| 08 | CatBoost symmetric oblivious trees | Every node at a given depth uses the same split. This restricts the tree structure, acts as regularization, fixes the number of leaves at `2^depth`, and supports efficient inference. | `08_catboost_symmetric_oblivious_trees.mp4` |
| 09 | Early stopping as model selection | Each boosting iteration is a candidate model. Early stopping selects the iteration with the best development performance, so repeated use of the same validation data must be treated as a selection risk. | `09_early_stopping_as_model_selection.mp4` |
| 10 | Governed models, constraints, and inspection | Monotonic constraints can encode trusted policy or domain relationships. Feature importance and SHAP support inspection, but neither should be confused with a causal explanation. | No standalone video found |
| 11 | CatBoost heterogeneous text and embedding features | CatBoost can combine numerical, categorical, text, and embedding features, but richer inputs still require a prediction-time hypothesis, leakage checks, baseline comparison, and complexity justification. | No standalone video found |

### Video lessons

| Lesson | Video file | Duration |
|---|---|---:|
| Prediction-time data contract and semantic features | `00_prediction_time_data_contract_and_semantic_features.mp4` | 7:56 |
| Feature engineering as a business hypothesis | `01_feature_engineering_as_business_hypothesis.mp4` | 11:13 |
| Ablation studies over feature storytelling | `02_ablation_studies_over_feature_storytelling.mp4` | 9:09 |
| Coding walkthrough: advanced feature engineering | `02_code_advanced_feature_engineering.mp4` | 15:58 |
| Production-safe categorical encoding | `03_production_safe_categorical_encoding.mp4` | 11:28 |
| CatBoost ordered category statistics | `04_catboost_ordered_category_statistics.mp4` | 11:20 |
| Ordered boosting and prediction shift | `05_ordered_boosting_and_prediction_shift.mp4` | 8:51 |
| Tree ensembles and automatic interactions | `06_tree_ensembles_automatic_interactions.mp4` | 8:48 |
| CatBoost native missing-value splits | `07_catboost_native_missing_value_splits.mp4` | 6:41 |
| CatBoost symmetric oblivious trees | `08_catboost_symmetric_oblivious_trees.mp4` | 6:32 |
| Early stopping as model selection | `09_early_stopping_as_model_selection.mp4` | 5:41 |
| **Module 02 total** |  | **1:43:37** |

### Module outcome

By the end of Module 02, the learner should be able to propose features as testable business hypotheses, evaluate feature families with ablation studies, choose a safe categorical strategy, explain the main CatBoost mechanisms, preserve semantic meaning in missing and sentinel values, and inspect model behavior under governance constraints.

## Recommended learning sequence

For each module, use the concept notes as the conceptual reference, watch the matching videos, and then run the corresponding notebook:

| Sequence | Notebook | Focus |
|---:|---|---|
| 1 | `00_course_setup_and_dataset.ipynb` | Setup, dataset, prediction contract, leakage, validation, and baseline |
| 2 | `01_gradient_boosting_fundamentals.ipynb` | Boosted-tree mechanics, probabilities, capacity, and validation |
| 3 | `02_advanced_feature_engineering.ipynb` | Safe features, ablations, categorical handling, CatBoost, and inspection |

The dependency order matters: Module 00 defines the legal data and evaluation contract; Module 01 explains the model family used later; Module 02 improves representations without abandoning the contracts established earlier.

## Completion standard for these modules

A learner has completed this foundation when they can explain not only which model or feature performed best, but also:

- what decision the model supports;
- when the prediction is made;
- why every feature is available and semantically valid at that moment;
- how preprocessing and feature construction avoid leakage;
- what evidence justifies added complexity;
- how probabilities become decisions;
- how validation and test data are protected from repeated selection; and
- which model behaviors require inspection, constraints, or monitoring.
