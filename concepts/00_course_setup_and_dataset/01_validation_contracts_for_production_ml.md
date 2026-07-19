This is one of the biggest mindset shifts from building models in notebooks to building **production ML systems**.

Most junior data scientists think validation means **"does my DataFrame look okay?"** A senior data scientist thinks **"can I trust every prediction this system will ever make?"**

The phrase **"validate contracts at two levels"** means you should protect your system both **when data first arrives** and **every time the model is used**.

---

# First level: Ingestion Contract

Imagine your bank marketing dataset arrives every morning.

Before you even train a model, you want to ask:

> "Did I receive the data I expected?"

These checks validate the **dataset itself**.

Examples:

| Check | Why |
|--------|-----|
| Expected columns exist | Prevent schema changes |
| Row count reasonable | Detect partial loads |
| No duplicate primary keys | Detect ETL bugs |
| Target only contains yes/no | Prevent label corruption |
| Age between 18 and 100 | Detect impossible values |
| Job category is known | Detect new categories |
| Missing rate < expected | Detect upstream failures |

Example:

Yesterday

```
rows = 45,231
```

Today

```
rows = 4,812
```

Nothing crashed.

The CSV loads perfectly.

A junior DS happily trains.

A senior DS immediately stops the pipeline.

Why?

Maybe yesterday's ETL failed.

Training on 10% of the data would silently degrade the model.

---

Another example:

Yesterday:

```
target
yes
no
```

Today:

```
target
Yes
No
UNKNOWN
```

The pipeline may still run.

But the labels are now inconsistent.

A contract catches this immediately.

---

# Second level: Pipeline Contract

Now suppose the data passed ingestion.

You split into train/validation.

You fit preprocessing.

You deploy.

Weeks later...

Production sends a prediction request.

Now you ask a different question:

> "Can my model safely score THIS batch?"

These checks happen every single time the model is used.

---

Examples:

### Feature existence

Model expects

```
age
balance
job
housing
loan
```

Production sends

```
age
balance
housing
loan
```

Missing:

```
job
```

Do you predict anyway?

No.

Fail immediately.

---

### Feature type

Training

```
balance : float
```

Production

```
balance : "unknown"
```

That should fail.

---

### Feature range

Training

```
age
18-95
```

Production

```
age = 540
```

Clearly something is wrong upstream.

---

### Category validation

Training

```
job

admin
student
management
retired
```

Production suddenly has

```
job = astronaut
```

Should the model accept it?

Maybe.

Maybe not.

At minimum you should log it.

If 30% of customers suddenly become "astronaut," something upstream changed.

---

### Missing values

Training

```
income missing = 2%
```

Production

```
income missing = 87%
```

The model technically still predicts.

But can you trust those predictions?

Probably not.

---

# Why separate ingestion from pipeline?

Because they solve different problems.

### Ingestion protects your data platform

```
Raw Data
     ↓
Validation
     ↓
Storage
```

Questions:

- Is the file complete?
- Is the schema correct?
- Are labels valid?
- Did ETL succeed?

---

### Pipeline validation protects your ML system

```
Features
      ↓
Validation
      ↓
Model
      ↓
Prediction
```

Questions:

- Can the model score this?
- Are features compatible?
- Has drift occurred?
- Is preprocessing valid?

---

# Real Production Architecture

A typical production flow looks like this:

```
Database
      │
      ▼
ETL Pipeline
      │
      ▼
Ingestion Validation
      │
      ▼
Feature Store
      │
      ▼
Feature Validation
      │
      ▼
Model
      │
      ▼
Prediction
      │
      ▼
Monitoring
```

Notice there are **multiple validation layers**, not just one.

---

# Why "fail fast" instead of fixing data automatically?

This surprises many people.

Suppose production sends:

```
age = -999
```

A junior engineer might think:

> "I'll replace it with the median."

A senior engineer asks:

> **Why is age suddenly -999?**

Maybe:

- ETL broke
- Database schema changed
- CSV parser failed
- Nulls were encoded incorrectly

Silently "cleaning" the data hides the real problem. Instead, critical contract violations should cause the pipeline to **fail or quarantine the affected data** so the upstream issue gets investigated. Automatic fixes are appropriate only for issues you've explicitly designed for (for example, expected missing values handled by a trained imputer), not for unexpected contract breaches.

---

# What libraries help with this?

Modern ML teams often formalize these checks using data validation frameworks:

- **Pandera** — validates pandas DataFrames using schemas.
- **Great Expectations** — defines and runs data quality expectations in ETL and ML pipelines.
- **TensorFlow Data Validation (TFDV)** — analyzes schemas and detects anomalies in TensorFlow pipelines.
- **Pydantic** — validates structured Python objects and API inputs before they reach the model.

These tools let you define contracts declaratively and automate validation in CI/CD and production.

---

# The senior mindset

The most important shift is this:

| Junior thinking | Senior thinking |
|-----------------|-----------------|
| "Can I train the model?" | "Can I trust the data?" |
| "The notebook runs." | "The pipeline is reliable." |
| "I'll clean bad values." | "Why did bad values appear?" |
| "The model didn't crash." | "The model is making trustworthy predictions." |
| "Validation is preprocessing." | "Validation is a production reliability guarantee." |

Ultimately, high-performing ML systems are rarely distinguished by a slightly better algorithm. They are distinguished by **reliable data contracts** that ensure the model is always trained and served on data that matches the assumptions under which it was built. That's why experienced ML engineers and senior data scientists often spend as much time on data validation and monitoring as they do on model development.