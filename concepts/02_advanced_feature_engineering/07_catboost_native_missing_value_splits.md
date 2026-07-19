This is one of CatBoost's more subtle advantages, and it's often misunderstood. The key idea is that **"missing" is not always the same as "unknown."** Sometimes the absence of a value is itself informative.

## Native numeric missing-value splits

Most machine learning algorithms cannot work with `NaN` values directly. Before training, you typically have to impute them.

For example:

| Customer | Income |
|----------|--------:|
| A | $45,000 |
| B | NaN |
| C | $82,000 |

A typical preprocessing pipeline might do:

```text
NaN → median income ($58,000)
```

or

```text
NaN → mean income
```

The model then never knows that Customer B originally had missing income.

---

## What CatBoost does differently

CatBoost keeps the missing value during training.

Instead of replacing it, the algorithm considers splits like

```text
Income is Missing?
        /        \
      Yes        No
```

or

```text
Income < 50,000 ?
```

where missing values are treated as their own possible branch.

Conceptually, the tree may learn

```
                 Income Missing?
                /                \
             Yes                  No
       Higher risk          Income < 50k?
```

So the model can discover that

> "Customers who didn't report their income behave differently."

without requiring manual imputation.

---

## Why can missingness be predictive?

Suppose you're predicting loan default.

| Income | Default |
|--------:|---------|
| 45k | No |
| 52k | No |
| 67k | No |
| NaN | Yes |
| NaN | Yes |
| NaN | Yes |

The missing value itself carries information.

Perhaps customers hiding income are riskier.

Replacing

```
NaN → 55k
```

would destroy that signal.

CatBoost can preserve it.

---

## Another real-world example

Imagine an online application.

| Feature | Meaning |
|----------|---------|
| Years at current job | Missing |

Missing might mean

- unemployed
- self-employed
- retired
- refused to answer
- new applicant

Those situations often have different behavior than simply having "5 years."

Again, the absence is informative.

---

# But this does **NOT** mean every special value should become NaN

This is the important senior-level distinction.

Consider the Bank Marketing dataset.

```
pdays
```

represents

> Days since the customer was previously contacted.

The dataset defines

```
pdays = -1
```

as

> Customer has never been contacted before.

This is **not missing data.**

It is a legitimate business state.

```
-1
```

has a specific semantic meaning.

Replacing it with

```
NaN
```

throws away information.

---

## Better feature engineering

Instead of pretending it is missing, separate the business concepts.

```python
previously_contacted = (pdays != -1).astype(int)

days_since_previous_contact = (
    pdays.replace(-1, np.nan)
)
```

Now you have

Feature 1

```
previously_contacted
```

```
0 = never contacted
1 = contacted before
```

Feature 2

```
days_since_previous_contact
```

```
NaN
```

only means

> "No previous contact exists."

Now the model learns two independent signals.

---

## Why is this better?

Suppose the data look like

| Previously Contacted | Days Since Contact | Subscribe |
|----------------------|-------------------:|-----------|
| No | NaN | Low |
| Yes | 7 | High |
| Yes | 30 | Medium |
| Yes | 300 | Low |

The model can separately learn

- whether the customer has ever been contacted
- how recent that contact was

Those are different business hypotheses.

---

# Senior mindset

A useful principle is:

> **Use native missing-value handling only when the value is genuinely unknown or unavailable. Do not use it to hide business states.**

Ask yourself:

> **"Does this value mean 'we don't know,' or does it mean something specific about the business process?"**

If it means something specific—like

- never contacted
- account not opened
- customer opted out
- no previous purchase

then model that state explicitly.

If it truly means

- sensor failed
- customer skipped the question
- data unavailable
- value not recorded

then CatBoost's native missing-value handling can often exploit the predictive information contained in the missingness itself.

This distinction is one of the habits that separates feature engineering from simply preprocessing data. The goal is not to eliminate missing values—it is to preserve the semantics of the data while giving the model the right representation to learn from.
