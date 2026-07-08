# Data

## Primary dataset

This course uses **UCI Bank Marketing**, DOI
[`10.24432/C5K306`](https://doi.org/10.24432/C5K306), created by S. Moro,
P. Rita, and P. Cortez. UCI distributes it under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The notebooks download the documented archive from:

`https://archive.ics.uci.edu/static/public/222/bank+marketing.zip`

The UCI bundle contains a nested `bank.zip`; only its documented
`bank-full.csv` member is extracted to `data/raw/bank-full.csv`. The raw file
is ignored by Git. No code from either archive is executed. Delete the local
CSV to repeat the download.

## Target and prediction moment

`y` records whether a contacted client subscribed to a term deposit. The
course frames the model as a **pre-contact prioritization** system. Therefore
`duration` (the duration of the current call) is unavailable at prediction
time and is excluded from production features. Notebook 00 demonstrates why
including it would produce an unrealistically optimistic result.

`unknown` is a documented category, not a parsed null. `pdays == -1` means no
previous contact and is handled as a sentinel with an explicit indicator.

## Known limitations

- The data describe one Portuguese bank and historical campaigns; portability
  to other institutions or periods is not established.
- There is no stable customer identifier, so repeated-client groups cannot be
  reconstructed and grouped validation cannot be guaranteed.
- The older `bank-full.csv` is ordered but lacks a year, so a defensible true
  temporal split cannot be reconstructed from `month` and `day` alone.
- Subscription is an observed outcome after contact, not a causal estimate of
  the incremental effect of calling a client.
- Sensitive and socioeconomic attributes require governance beyond predictive
  performance before real deployment.

Large datasets and generated model artifacts are intentionally not committed.
