# Context

## Glossary

### Prediction moment

The instant immediately before a marketing call is placed. Only information
known by this instant may be used by the production model.

### Development set

The labeled subset used for fitting and cross-validated model selection.

### Validation set

The labeled subset used after model selection to choose an operating threshold
and compare a small number of finalized candidates. It is not the test set.

### Test set

The sealed labeled subset used once in notebook 09 for final evaluation. It is
never used for feature selection, hyperparameter tuning, or threshold choice.

### Business cost

An explicit teaching utility where a false positive costs 1 unit and a false
negative costs 5 units. These values are illustrative assumptions, not claims
about the original bank.

