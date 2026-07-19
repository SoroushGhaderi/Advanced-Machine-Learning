# Data Governance and Feature Contracts

This is one of the biggest differences between a junior and a senior data scientist. Most ML failures in production are **not** caused by a poor algorithm—they're caused by poor data governance.

Let's unpack each concept.

---

# Why data governance matters

A typical junior workflow looks like this:

```
CSV
 ↓
Pandas
 ↓
Train XGBoost
 ↓
95% AUC
```

A senior workflow looks more like this:

```
Business decision
        ↓
Define prediction moment
        ↓
Locate trustworthy data
        ↓
Verify contracts
        ↓
Feature engineering
        ↓
Model training
        ↓
Deployment
        ↓
Monitoring
```

Notice that modeling is only one step.

The hardest questions are usually:

- Can this feature be trusted?
- Will this feature exist in production?
- Who owns this data?
- When is it updated?
- What happens if it disappears?

Those are governance questions.

---

# 1. Data Catalog

Think of a data catalog as **Google Maps for company data**.

It answers questions like:

> Where does this dataset come from?

> Who owns it?

> Can I trust it?

| Dataset | Owner | Updated | Source | Purpose |
|----------|----------|----------|-----------|-----------|
| Customers | CRM Team | Daily | Salesforce | Customer master |
| Marketing Calls | Marketing | Hourly | Dialer | Campaign tracking |
| Transactions | Finance | Real-time | Payment System | Revenue |

Notice that this says almost nothing about individual columns.

Instead it describes the **asset**.

---

# 2. Data Dictionary

A data dictionary explains every column.

| Column | Type | Meaning |
|----------|------|----------|
| age | integer | Customer age |
| job | categorical | Occupation |
| balance | float | Current bank balance |
| duration | integer | Call duration in seconds |

This is what most people think documentation is.

But for ML, this isn't enough.

---

# 3. Feature Contract

This is where senior ML starts.

A feature contract extends a data dictionary with production guarantees.

Instead of

```
age
integer
```

you specify something like

| Property | Value |
|-----------|---------|
| Name | age |
| Type | integer |
| Nullable | No |
| Range | 18–100 |
| Available before prediction? | Yes |
| Refresh frequency | Daily |
| Owner | CRM Team |
| Missing value policy | Reject row |
| Validation | age > 17 |

Now you're documenting not only *what* the feature is, but *how it behaves operationally*.

---

# The most important addition: Feature availability

Suppose your model predicts:

> Will a customer subscribe before we call them?

Then consider this feature:

```
duration
```

Technically:

- integer
- between 0 and 6000

Everything looks fine.

But...

When is it known?

```
Customer picks up phone
        ↓
Conversation happens
        ↓
Call ends
        ↓
Duration becomes available
```

Your prediction happened **before** the phone call.

Therefore

```
duration
```

violates the feature contract.

Not because its values are wrong.

Because its **availability timing** is wrong.

This is why feature contracts explicitly include:

```
Available at prediction time?
YES / NO
```

---

# Another example

Imagine predicting fraud at purchase time.

Available:

```
Customer age
Account age
Country
Merchant
Time of day
```

Unavailable:

```
Chargeback filed
Investigation completed
Fraud confirmed
```

All of those occur later.

If they appear in training, your model is cheating.

---

# Validation Rules

Senior teams validate every incoming batch automatically.

For example:

```
age

Must be integer
18 ≤ age ≤ 100
Missing < 2%
```

```
balance

Must be numeric
No NaN
No infinity
```

```
job

Allowed values:
admin
technician
student
management
...
```

If validation fails:

```
STOP PIPELINE
```

instead of silently replacing bad values with zeros or means.

This prevents bad data from reaching production.

---

# Schema vs Semantics

This distinction is subtle but very important.

Schema answers:

```
Is this an integer?
```

Semantics answers:

```
Does this integer actually mean customer age?
```

Example:

```
age = 500
```

Schema:

✔ integer

Semantics:

✘ impossible

Another example:

```
balance = -500
```

Schema:

✔ float

Semantics:

Depends.

Negative balances may be perfectly valid in a banking system (e.g., overdrafts), so whether this is acceptable depends on the business definition—not just the data type.

Senior engineers validate **both** schema and semantics.

---

# Refresh Frequency

Another overlooked property.

Suppose:

```
Income
```

updates

```
once per month
```

but

```
Transactions
```

update

```
every second
```

Your production model must know this.

Otherwise you may assume a feature is current when it's actually weeks old.

---

# Ownership

Every feature should have an owner.

```
age

Owner:
CRM Team
```

```
balance

Owner:
Core Banking
```

Why?

Because when

```
balance suddenly becomes NULL
```

someone must be responsible for fixing it.

Without ownership:

Everyone assumes someone else will fix it.

Nobody does.

---

# Safe-use Conditions

This is often missing from beginner projects.

Example:

```
balance

Do not use for customers under 18.

Unavailable for legacy accounts.

Unavailable during maintenance window.

Delayed by 24 hours after migration.
```

These conditions determine when a feature is reliable.

---

# Putting it all together

A mature feature contract might look like this:

| Property | Value |
|----------|--------|
| Feature | balance |
| Type | float |
| Nullable | No |
| Allowed range | -100000 to 10000000 |
| Owner | Banking Platform |
| Refresh | Hourly |
| Available before prediction | Yes |
| Validation | Numeric, finite, non-null |
| Missing action | Reject scoring request |
| Monitoring | Alert if missing rate > 0.5% |

Notice how little of this is about machine learning. It's about making the data dependable.

---

## Why senior data scientists care so much about this

A common misconception is that seniority comes from knowing more algorithms. In practice, experienced data scientists often spend more time thinking about the *data lifecycle* than the model itself.

A rough progression looks like this:

| Junior focus | Senior focus |
|--------------|--------------|
| Which model has the highest AUC? | Can this model be trusted in production? |
| Feature engineering | Feature governance |
| Hyperparameter tuning | Data quality and feature availability |
| Model performance | End-to-end decision system reliability |
| Notebook experiments | Reproducible, monitored production pipelines |

A senior data scientist knows that a model with an AUC of **0.84** running on reliable, well-governed features will almost always create more business value than a model with an AUC of **0.87** that occasionally trains on stale data, leaks future information, or breaks when upstream schemas change.

That's why organizations with mature ML practices invest heavily in **data catalogs, feature contracts, validation, monitoring, and ownership**. These disciplines make machine learning systems reproducible, maintainable, and trustworthy long after the initial model is deployed.
