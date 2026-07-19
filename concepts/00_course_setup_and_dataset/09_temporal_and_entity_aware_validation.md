This is one of the biggest differences between an academic ML workflow and a production ML workflow.

The statement

> **"Stratification is not a substitute for temporal or entity-aware validation."**

means that **preserving the class ratio is often much less important than preserving the way data will actually arrive in production.**

Let's unpack it.

---

# What stratification actually does

Suppose you have a churn dataset.

|Customer|Month|Churn|
|---------|-----|-----|
|A|Jan|No|
|B|Jan|Yes|
|C|Feb|No|
|D|Feb|No|
|...|...|...|

A stratified split only guarantees something like

- Train: 10% churn
- Validation: 10% churn

It says nothing about

- when observations occurred
- whether the same customer appears multiple times
- whether future information leaks backwards

It preserves **label distribution**, nothing more.

---

# Why this can be misleading

Imagine you're building a model today.

Today is:

```
January 2025
```

The model will predict customers in

```
February 2025
March 2025
April 2025
```

Production always looks like

```
Past  ---> Future
```

Yet a random split creates

```
January
February
March
April

↓

Random shuffle

Train:
Jan Feb Apr Mar Jan Feb

Validation:
Jan Mar Apr Feb
```

Now training has already "seen the future."

This is **not leakage** in the classic sense—you're not directly using future labels—but it **does create an unrealistically easy prediction task** because the train and validation sets come from nearly identical distributions.

---

# Temporal drift

Real businesses change.

Maybe

- prices change
- competitors launch products
- inflation changes spending
- customer behavior evolves
- marketing campaigns change

The distribution becomes

```
2023  --------> 2024 ---------> 2025

Customer behavior slowly changes
```

If training includes pieces of 2025 while validating on other pieces of 2025,

the model looks amazing.

But in reality you'll train on

```
2024
```

and predict

```
2025
```

Those are different distributions.

A proper evaluation mirrors that.

```
Train:
2022-2024

Validate:
2025
```

Now you're measuring

> "Can this model survive tomorrow?"

instead of

> "Can this model interpolate yesterday?"

---

# Example

House prices.

Random split:

```
2020 houses
2021 houses
2022 houses

↓

Shuffle

Train:
2020
2021
2022

Validation:
2020
2021
2022
```

Easy.

The market conditions are shared.

---

Real deployment

Train

```
2020
2021
```

Predict

```
2022
```

Interest rates doubled.

Prices shifted.

The model suddenly performs much worse.

Random CV completely hid this problem.

---

# Entity leakage

This is even more common.

Imagine a medical dataset.

```
Patient A
Visit 1

Patient A
Visit 2

Patient A
Visit 3

Patient B
Visit 1

Patient C
Visit 1
```

Random splitting might produce

Training

```
Patient A Visit 1
Patient A Visit 2
Patient B
```

Validation

```
Patient A Visit 3
Patient C
```

The model has effectively already learned Patient A.

It recognizes

- age
- medications
- history
- lab values

Validation becomes artificially easy.

---

The same happens in

- recommender systems
- banking
- insurance
- fraud
- marketing
- healthcare

Whenever the same entity appears multiple times.

---

# Proper solution

Split by entity.

Instead of

```
Rows
```

split

```
Customers
```

Train

```
Customer A
Customer B
Customer C
```

Validation

```
Customer D
Customer E
```

Now the model must generalize to unseen customers.

That's production.

---

# Time + Entity

Many industrial datasets require both.

Suppose

```
Customer A
Jan

Customer A
Feb

Customer B
Jan

Customer B
Feb
```

A production split becomes

Train

```
Customers
A
B

Months
Jan
```

Validation

```
Customers
A
B

Months
Feb
```

or even

Train

```
Old customers
Old months
```

Validate

```
New customers
Future months
```

depending on the deployment scenario.

---

# Why Kaggle often uses random splits

Many benchmark datasets are i.i.d. (independent and identically distributed), meaning the benchmark assumes the training and test data come from the same underlying distribution.

Random stratified cross-validation estimates performance well under that assumption.

Production data rarely satisfies this assumption because:

- distributions drift over time,
- users return repeatedly,
- products evolve,
- business processes change.

This is why a model that scores **0.92 AUC** on random CV may fall to **0.80 AUC** after deployment.

---

# Senior data scientists ask a different question

A junior practitioner often asks:

> "How do I split the data?"

A senior practitioner asks:

> **"What data will be available when this model is actually making predictions?"**

The validation strategy should recreate that future prediction environment as closely as possible.

---

# Rule of thumb

Choose your validation strategy based on the deployment scenario:

| Deployment scenario | Recommended validation |
|---------------------|------------------------|
| Independent observations with no temporal or entity structure | Stratified K-Fold |
| Forecasting, demand prediction, stock prices, customer behavior over time | Time-based split / rolling-window validation |
| Multiple rows per customer, patient, device, merchant, etc. | Group-based validation (e.g., GroupKFold) |
| Time series with repeated entities (customers over time, IoT devices, sensors) | Group + time-aware validation |

A useful mindset is that **the validation set is a simulator of production**. If the simulator doesn't resemble how the model will actually be used, the reported performance can be overly optimistic—even if the code is technically correct.