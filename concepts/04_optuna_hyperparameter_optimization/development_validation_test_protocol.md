# Development, Validation, and Test Protocol

**Notebook:** `04_optuna_hyperparameter_optimization.ipynb`

Hyperparameter search belongs in the development set and its cross-validation folds. A validation set may support limited finalist comparison, threshold choice, calibration, or business-cost optimization. The test set is sealed until the workflow is frozen and is evaluated once.

Repeatedly choosing changes from test performance turns the test set into another validation set and overfits its noise.
