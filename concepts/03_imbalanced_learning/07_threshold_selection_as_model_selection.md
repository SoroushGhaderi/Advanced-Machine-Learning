This is one of the most subtle—and most important—ideas in applied machine learning. Many practitioners understand that **you shouldn't train on the test set**, but fewer realize that **using the test (or validation) set to choose a threshold is also a form of training**.

The key insight is:

> **Every decision you make after looking at evaluation data is a learning step.**

Changing a threshold from 0.50 to 0.37 because validation looked better is not changing model parameters, but it *is* adapting your system to that dataset.

---

# The three phases of ML

A useful senior mental model is to think of three completely different activities.

| Phase | Purpose | Data role |
|--------|----------|-----------|
| Learning | Learn parameters | Training data |
| Model selection | Choose among alternatives | Validation / CV |
| Final evaluation | Estimate future performance | Test |

The mistake occurs when these phases become mixed together.

---

# What exactly is threshold selection?

Suppose your model outputs probabilities.

| Customer | Probability |
|------------|------------|
| A | 0.96 |
| B | 0.82 |
| C | 0.74 |
| D | 0.61 |
| E | 0.44 |
| F | 0.29 |

The model itself has already finished learning.

Threshold selection asks:

> "At what probability should we intervene?"

Maybe

```
0.50
```

Maybe

```
0.32
```

Maybe

```
0.81
```

Nothing about the model changed.

Only the decision policy changed.

---

# But the threshold is still a learned parameter

This surprises many people.

Suppose you try

```
0.30
0.35
0.40
0.45
0.50
0.55
...
0.90
```

on the validation set.

Eventually you discover

```
Threshold = 0.43

F1 = 0.791
```

Looks great.

But why is it great?

Maybe because

- your model is good

or

- that particular validation split happened to favor 0.43

You cannot know.

You have optimized to that validation data.

That means the threshold itself has become another fitted hyperparameter.

---

# Threshold is just another hyperparameter

People recognize these as hyperparameters:

```
max_depth

learning_rate

C

gamma

subsample
```

But few recognize

```
decision_threshold
```

belongs in exactly the same category.

From an optimization perspective,

```
Choose learning rate

↓

Measure validation

↓

Keep best
```

is mathematically identical to

```
Choose threshold

↓

Measure validation

↓

Keep best
```

Both are optimization loops.

---

# Why repeatedly using validation is dangerous

Imagine validation contains only 400 positive examples.

Random sampling noise exists.

Perhaps

```
Threshold 0.43

Precision = 0.721
```

while

```
Threshold 0.44

Precision = 0.715
```

That tiny difference may be pure randomness.

If you always pick whichever looks slightly better,

you're fitting to noise.

This is exactly what overfitting means.

Only now you're overfitting

not the model

but the decision rule.

---

# Validation slowly becomes training

Imagine this workflow.

```
Train model

↓

Evaluate validation

↓

Threshold 0.50
```

You notice recall is low.

So you change to

```
0.35
```

Validation improves.

Great.

Then you notice precision is too low.

So you change

```
0.41
```

Better.

Then business wants

```
higher recall
```

So

```
0.38
```

Then calibration

```
0.42
```

Eventually you've looked at validation dozens of times.

Although weights never changed,

your whole pipeline has become specialized to that validation set.

Validation is no longer independent.

It has effectively become another training dataset.

---

# This is exactly why we need a test set

The test set should answer only one question:

> **If I deployed everything exactly as it is today, how well would it work?**

Everything means everything:

- preprocessing
- feature engineering
- sampler
- model
- calibration
- threshold
- business rules

Nothing should remain undecided.

Otherwise the test set becomes another optimization target.

---

# The proper production workflow

A senior workflow looks like this:

```
Raw data
      │
      ▼
Development
```

Inside development you may optimize:

```
Model

↓

Sampler

↓

Hyperparameters

↓

Calibration

↓

Threshold
```

using nested cross-validation or out-of-fold predictions.

Only after **all decisions are frozen** do you move on.

```
Lock pipeline
```

Now evaluate once.

```
Validation
```

If validation agrees with development,

lock everything again.

Only then

```
Test

↓

Report final numbers
```

The test set is opened exactly once.

---

# Why nested CV is so powerful

Suppose you're comparing:

- Logistic Regression
- XGBoost
- CatBoost

For each one you also compare

- class weights
- SMOTE
- SMOTENC

For each you compare

- calibration methods

For each you compare

- thresholds

This is no longer "choosing a threshold."

You're selecting an entire deployment policy.

Nested CV keeps all of these choices inside the inner optimization loop.

The outer folds evaluate the *fully selected pipeline* on unseen data, providing a much less biased estimate of how well your entire model-selection process generalizes.

---

# Out-of-fold predictions are ideal for threshold selection

Suppose you have 5-fold CV.

For each fold:

```
Train on 80%

↓

Predict remaining 20%
```

Eventually every observation receives a prediction from a model that never saw it.

These are **out-of-fold (OOF) predictions**.

```
Entire dataset

↓

OOF probabilities

↓

Threshold optimization
```

This is far superior to optimizing the threshold on the same data used to fit the model because every probability is generated independently of the observation being predicted.

---

# A concrete example

Suppose after OOF prediction you compute:

| Threshold | Cost |
|-----------|------|
| 0.20 | \$18,100 |
| 0.25 | \$16,500 |
| 0.30 | \$15,200 |
| 0.35 | \$14,700 |
| 0.40 | \$15,400 |

You select

```
Threshold = 0.35
```

Now freeze it.

Never touch it again.

Now evaluate on validation.

Suppose validation produces

```
Cost = $15,100
```

Good.

Now freeze everything.

Finally

```
Test

↓

Cost = $15,300
```

That number is believable because the test data never influenced **any** decision.

---

# Why this matters even more than model choice

Imagine two scenarios.

### Team A

They tune:

- model
- sampler
- calibration
- threshold

using proper nested CV.

Their test estimate is unbiased.

### Team B

They keep checking validation after every change:

- threshold
- calibration
- feature engineering
- cost function
- business rule

Eventually validation looks fantastic.

Then deployment happens.

Performance drops dramatically.

Why?

Because validation quietly became another training dataset.

This phenomenon is sometimes called **evaluation leakage** or **selection bias**: information from the evaluation data leaks into the development process through repeated decisions, even though the model's weights are never directly trained on that data.

---

# The senior mindset

A junior practitioner often thinks:

> "The model learns from the training data."

A senior practitioner thinks:

> **The entire ML pipeline learns from every dataset that influences a decision.**

That pipeline includes:

- feature engineering
- preprocessing
- resampling strategy
- hyperparameters
- probability calibration
- decision threshold
- business policy

Every one of these choices should be selected using development data, confirmed on validation if appropriate, and then evaluated exactly once on a sealed test set. Once you view threshold selection this way—as another optimization problem rather than a post-processing tweak—the need to protect your evaluation data becomes much clearer.