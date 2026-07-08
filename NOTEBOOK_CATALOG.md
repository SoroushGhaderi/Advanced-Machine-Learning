# Notebook Catalog

This repository contains a guided sequence of notebooks for advanced machine learning practice. The `notebooks/` folder holds the main English version, and `notebooks_farsi/` holds the Persian translation of the same material.

## 00 - Course Setup and Dataset

- File: `notebooks/00_course_setup_and_dataset.ipynb`
- Definition: Introduces the project environment, the dataset, and the initial data catalog.
- Why we created it: To make sure everyone starts from the same setup and understands what data is being used before any modeling begins.
- How it helps us: It reduces confusion, makes the dataset easier to find and govern, and creates a reliable base for all later notebooks.
- Key information considered:
  - Dataset origin and structure
  - Data catalog and field definitions
  - Initial assumptions about the problem
  - Basic environment and package setup

## 01 - Gradient Boosting Fundamentals

- File: `notebooks/01_gradient_boosting_fundamentals.ipynb`
- Definition: Explains the core ideas behind gradient boosting models and how they learn from weak learners.
- Why we created it: To build intuition before moving into more advanced feature engineering and optimization.
- How it helps us: It gives a strong baseline model family for tabular machine learning problems and helps explain later performance improvements.
- Key information considered:
  - How boosting reduces error iteratively
  - Model training and validation flow
  - Common mistakes and leakage risks
  - When gradient boosting is a good fit

## 02 - Advanced Feature Engineering

- File: `notebooks/02_advanced_feature_engineering.ipynb`
- Definition: Covers more advanced ways to transform raw data into model-friendly features.
- Why we created it: To improve predictive power beyond the baseline by using better representations of the input data.
- How it helps us: Well-designed features often improve model quality more than small algorithm changes.
- Key information considered:
  - Categorical feature handling
  - Preventing target leakage
  - Practical feature engineering workflow
  - Feature construction choices and tradeoffs

## 03 - Imbalanced Learning

- File: `notebooks/03_imbalanced_learning.ipynb`
- Definition: Focuses on modeling problems where one class appears much less often than the other.
- Why we created it: Many real-world ML tasks are imbalanced, so accuracy alone is not enough.
- How it helps us: It teaches us how to choose better metrics and sampling strategies for rare-event prediction.
- Key information considered:
  - What imbalance changes in training and evaluation
  - Suitable metrics for skewed targets
  - Resampling and class weighting ideas
  - Risks of misleading performance estimates

## 04 - Optuna Hyperparameter Optimization

- File: `notebooks/04_optuna_hyperparameter_optimization.ipynb`
- Definition: Shows how to automate hyperparameter search with Optuna.
- Why we created it: To systematically improve models instead of tuning parameters by hand.
- How it helps us: It makes experimentation faster, more reproducible, and easier to compare across runs.
- Key information considered:
  - Search space design
  - Objective function definition
  - Study setup and optimization strategy
  - Avoiding overfitting during tuning

## 05 - Ensemble Learning

- File: `notebooks/05_ensemble_learning.ipynb`
- Definition: Introduces combining multiple models to improve robustness and performance.
- Why we created it: To show how multiple learners can work together better than a single model.
- How it helps us: Ensembles often improve generalization and reduce variance in real-world ML systems.
- Key information considered:
  - Bagging, boosting, and stacking ideas
  - When ensemble methods help most
  - Tradeoffs in complexity and interpretability
  - Evaluation of combined models

## 06 - Anomaly Detection Extension

- File: `notebooks/06_anomaly_detection_extension.ipynb`
- Definition: Explores methods for detecting unusual or rare observations.
- Why we created it: To extend the course beyond standard supervised learning into rare-event and outlier-oriented tasks.
- How it helps us: It broadens the toolkit for fraud, fault, and incident detection use cases.
- Key information considered:
  - What counts as an anomaly
  - Detection strategy choices
  - Thresholding and false alarm tradeoffs
  - Validation when labels are limited

## 07 - End-to-End Production ML Project

- File: `notebooks/07_end_to_end_production_ml_project.ipynb`
- Definition: Walks through a complete machine learning workflow from problem framing to local data loading and production-oriented thinking.
- Why we created it: To connect the earlier concepts into a realistic project workflow.
- How it helps us: It shows how the pieces fit together in a production-style ML lifecycle rather than in isolation.
- Key information considered:
  - Problem framing and success criteria
  - Data loading and preparation
  - Workflow organization
  - Production concerns and project structure

## Notebook Relationship

- `00` establishes the dataset and setup.
- `01` builds the modeling foundation.
- `02` improves the input representation.
- `03` handles skewed outcomes.
- `04` tunes model performance more systematically.
- `05` combines models for stronger results.
- `06` extends the course to anomaly detection.
- `07` pulls the full workflow together in an end-to-end project.

