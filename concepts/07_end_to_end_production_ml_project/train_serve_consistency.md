# Train–Serve Consistency

**Notebook:** `07_end_to_end_production_ml_project.ipynb`

The fitted preprocessing and model artifact must be reused during inference. A single pipeline prevents notebook preprocessing from diverging from production preprocessing and makes the artifact reproducible.

Serving validation must check every prediction batch as a new dataset and reject features unavailable at the prediction moment.
