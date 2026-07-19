This is one of the most important—and most misunderstood—concepts in imbalanced learning. It's also one of the differences between someone who builds a model and someone who deploys one.

The key idea is:

> **Ranking and probability estimation are different tasks.**
>
> Resampling (SMOTE, oversampling, undersampling) often preserves ranking reasonably well, but it can distort probabilities.

Let's unpack why.

---

# What does a probability actually mean?

Suppose your model predicts

```text
Customer A: 0.70
```

A junior DS often interprets this as

> "There is a 70% chance this customer will churn."

That interpretation is only valid if the model is **well calibrated**.

Calibration means:

Among all customers receiving a prediction around **0.70**,

approximately **70% actually churn.**

Graphically:

```
Predicted 0.70
      │
      ▼

1000 customers

Expected:
700 churn
300 stay

Reality:
700 churn
300 stay
```

Then the model is calibrated.

---

# Ranking is a completely different property

Imagine another model:

```
Customer A   0.99
Customer B   0.95
Customer C   0.90
Customer D   0.80
Customer E   0.10
```

Suppose actual churn rates are

```
A yes
B yes
C yes
D yes
E no
```

The ordering is perfect.

ROC-AUC might be nearly 1.

But suppose reality is

```
0.99 group → 65% churn
0.95 group → 60%
0.90 group → 58%
0.80 group → 45%
```

The ranking is excellent.

The probabilities are terrible.

This model is **high discrimination, poor calibration.**

---

# Why does resampling break calibration?

Imagine the real world.

```
100,000 customers

99,000 non-churn
1,000 churn

Real prevalence = 1%
```

Now apply SMOTE.

Training data becomes

```
99,000 non-churn
99,000 synthetic/real churn

Training prevalence = 50%
```

Notice what happened.

The model is now learning from a world where

```
Positive class = 50%
```

instead of

```
Positive class = 1%
```

Those are fundamentally different probability distributions.

---

## Think like a Bayesian

The posterior probability is

\[
P(Y=1\mid X)
\]

which depends on

- likelihood
- prior probability

SMOTE changes the effective prior.

The model therefore learns

```
P_train(Y=1 | X)
```

not

```
P_real(Y=1 | X)
```

Those aren't identical.

---

# Simple numerical example

Suppose a feature is

```
Age > 60
```

Real data

```
1000 people

20 positives
980 negatives
```

Within this subgroup

```
10 positives
90 negatives

Probability = 10%
```

Now oversample positives.

Training becomes

```
100 positives
100 negatives
```

Inside the subgroup

```
50 positives
90 negatives

Probability ≈36%
```

The model now thinks

```
Age>60

≈36% chance
```

instead of

```
10%
```

The feature relationship still exists.

But the absolute probabilities changed.

---

# But isn't logistic regression estimating probabilities?

Yes.

**Only under its training distribution.**

Logistic regression estimates

\[
P(Y=1\mid X)
\]

assuming the observed sample represents the true population.

If you deliberately change

```
99:1
```

into

```
50:50
```

you violated that assumption.

The optimization is still mathematically correct.

The probabilities now refer to the altered distribution.

---

# What about class weights?

Many people think

> "Weights don't change the data, so probabilities stay valid."

Not necessarily.

Weighted logistic regression minimizes

\[
\sum_i w_i \cdot \text{LogLoss}_i
\]

Instead of

```
positive error = 1
negative error = 1
```

you may have

```
positive error = 99
negative error = 1
```

The optimizer is solving a different objective.

The learned coefficients shift.

The raw scores change.

Consequently, the predicted probabilities can become miscalibrated with respect to the real population, even though the training examples themselves were unchanged.

---

# Why boosting models are especially affected

Consider XGBoost.

With

```python
scale_pos_weight = 99
```

every positive residual is magnified.

Trees become much more aggressive at separating positives.

Excellent for ranking.

Not necessarily good for calibration.

The output

```
0.92
```

might correspond to

```
actual frequency = 0.55
```

---

# Why this matters in business

Imagine fraud detection.

Suppose

```
Model A

Fraud probability = 0.85
```

You investigate cases above

```
0.80
```

If probabilities are calibrated,

```
1000 investigations

850 frauds
```

You can estimate:

- staffing
- ROI
- expected recoveries
- budget

If calibration is poor,

```
1000 investigations

300 frauds
```

your operational planning can be far off, even if the model still ranks fraudsters near the top.

---

# Calibration methods

After training, we can learn a mapping

```
raw score
      │
      ▼
correct probability
```

instead of trusting the raw model output.

Common approaches are:

### Platt Scaling

Learns another logistic function

```
Raw score

↓

Logistic correction

↓

Probability
```

Works well with limited calibration data.

---

### Isotonic Regression

Learns a flexible monotonic curve

```
Raw score

↓

Flexible staircase

↓

Probability
```

Better with larger calibration datasets.

Can overfit if the calibration set is small.

---

# The senior workflow

A production workflow separates four distinct questions:

```
Training
     │
     ▼

Model

     │

Does it rank well?
        │
        ▼
ROC-AUC
PR-AUC

     │

Are probabilities accurate?
        │
        ▼
Calibration curve
Brier score
Log loss

     │

Need calibration?
        │
      yes
        │
        ▼
Platt / Isotonic

     │

Choose business threshold
        │
        ▼
Deploy
```

Notice the order:

1. **Learn a model** that separates classes well.
2. **Evaluate ranking** (ROC-AUC, PR-AUC).
3. **Evaluate calibration** on untouched data.
4. **Calibrate if needed** using a dedicated calibration set or out-of-fold predictions.
5. **Choose the business policy** (threshold, top-\(K\), expected-value optimization) using the calibrated probabilities.

A common mistake is to tune the threshold first and only later discover that a predicted probability of 0.80 actually corresponds to an observed event rate of 0.45. In that situation, the threshold was chosen on misleading probability estimates.

### A subtle but important point

Resampling does **not** automatically ruin calibration. Its effect depends on the model, the sampling strategy, and the learning algorithm. Some algorithms are more robust than others, and calibration may remain acceptable in some cases. The important principle is that **you should not assume probabilities remain calibrated after changing the class distribution or loss function**. Instead, verify calibration on untouched data and recalibrate if necessary.

This is why experienced practitioners treat **discrimination (ranking)** and **calibration (probability accuracy)** as two separate model properties. A model can be excellent at deciding *who is riskier than whom* while being poor at estimating *how risky they actually are*. The former is sufficient for ranking tasks; the latter is essential whenever probabilities drive expected-value calculations, budgeting, resource allocation, or automated decision-making.