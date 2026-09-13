# Capstone Rubric

Score each category from 0 to 4. Total: **24 points**.

| Category | 4 — Excellent | 3 — Competent | 2 — Partial | 1–0 — Insufficient |
|---|---|---|---|---|
| Correctness | Workflow runs end to end; claims match code and evidence; edge cases are tested. | Core workflow is correct with minor gaps. | Material errors or fragile assumptions remain. | Results are not reproducible or technically invalid. |
| Leakage prevention | Prediction moment is explicit; learned transforms/resampling/tuning stay inside development/CV; test is used once. | Main boundaries are correct with a small documentation gap. | One risky preprocessing or selection step is outside its boundary. | Test-driven iteration, target leakage, or contaminated CV invalidates results. |
| Evaluation and usefulness | Discrimination, calibration, thresholds, uncertainty, and explicit business costs/capacity are connected. | Multiple suitable metrics and an operational threshold are justified. | Metrics are mostly statistical with weak decision linkage. | Accuracy-only or unsupported usefulness claims. |
| Explainability and responsibility | Global/local explanations use the correct method; stability, dependence, non-causality, and limitations are communicated. | Correct explanations with major caveats stated. | Plots are present but weakly interpreted or unstable. | Attribution is incorrect, causalized, or absent. |
| Reproducibility and engineering | Pinned ranges, seeds, data checksum/source, pipeline artifact, reload test, schema validation, and failure checks are complete. | Re-runnable pipeline and artifact with most metadata. | Hidden state, manual preprocessing, or missing validation remains. | Cannot reproduce or safely score new data. |
| Monitoring and operations | Immediate and delayed-label signals, alert policy, investigation, retraining gates, ownership, and rollback are specified. | Sensible drift/performance plan and retraining criteria. | Generic dashboard without response workflow. | “Retrain on drift” or no monitoring plan. |

## Minimum acceptance gates

A submission cannot pass, regardless of total score, if it:

- uses `duration` in the pre-contact production model;
- fits an encoder, selector, or sampler on validation/test before evaluation;
- optimizes hyperparameters or thresholds on the test set;
- trains a stacking meta-model on in-sample base predictions;
- cannot reload the serialized pipeline and reproduce predictions; or
- reports causal or fraud conclusions unsupported by the dataset.

## Suggested submission

- Executed notebook or small notebook sequence with real outputs.
- Serialized pipeline plus reproducibility metadata (do not commit large data).
- One-page model-card update.
- Monitoring table with signals, windows, thresholds, owner, and response.
- Short reflection on one rejected approach and why it failed the rubric.

