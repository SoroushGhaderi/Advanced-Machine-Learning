# Validation and Drift for the Anomaly Extension

**Notebook:** `06_anomaly_detection_extension.ipynb`

The anomaly extension has no natural anomaly label, so contamination and thresholds are controlled assumptions rather than ordinary supervised targets. Validate feature schema, ranges, missingness, scoring stability, and alert volume; monitor drift after deployment.

An anomaly score is not automatically a business decision. Investigate whether alerts are actionable and whether the assumed contamination rate remains plausible.
