# Tuning Without Test Leakage

**Notebook:** `04_optuna_hyperparameter_optimization.ipynb`

Optuna trials, feature decisions, preprocessing choices, and model comparisons are learning decisions. They must use development data only. Evaluate the chosen candidate on the sealed test set after tuning, not during the search.

This preserves an honest estimate of future performance and makes experiments reproducible.
