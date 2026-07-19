# Production Data Contracts

**Notebook:** `07_end_to_end_production_ml_project.ipynb`

Before training or scoring, enforce contracts for required features, types, ranges, categories, missingness, freshness, availability, and ownership. Fail or quarantine unexpected violations rather than silently hiding upstream failures.

The model is one component of a governed system: database, ETL, validation, feature preparation, model, prediction, and monitoring each need explicit boundaries.
