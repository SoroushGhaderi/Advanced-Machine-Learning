This is one of the biggest mindset shifts from junior to senior machine learning.

Most people think there is only one question:

> "Is my classifier good?"

A senior data scientist realizes there are actually **three completely different optimization problems**, each owned by a different part of the ML system.

---

# Layer 1: Classification Performance

**Question:**

> If I choose a threshold, how many predictions are correct?

This is what nearly every introductory ML course focuses on.

Suppose your model outputs

| Customer | Probability |
|----------|------------:|
| A | 0.97 |
| B | 0.81 |
| C | 0.61 |
| D | 0.42 |
| E | 0.18 |

Now choose

Threshold = 0.5

Predictions become

```
A → Positive
B → Positive
C → Positive
D → Negative
E → Negative
```

Now you compute

- Precision
- Recall
- F1
- Accuracy
- Specificity
- Balanced Accuracy

Notice something important.

These metrics depend on **the threshold**, not just the model.

Change the threshold to 0.3

```
A Positive
B Positive
C Positive
D Positive
E Negative
```

The model hasn't changed.

Only the decision rule changed.

Yet

- Recall increases
- Precision decreases
- F1 changes
- Accuracy changes

Exactly the same model.

Different operational policy.

---

## Junior mistake

They say

> "Model B is better because Recall = 95%."

A senior immediately asks

> At what threshold?

because recall alone says nothing.

---

# Layer 2: Probability Quality

Now forget thresholds entirely.

Instead ask

> Are the probabilities themselves correct?

This is a completely different question.

Imagine two models.

### Model A

| Prediction | Reality |
|------------|---------|
|0.90|Positive|
|0.80|Positive|
|0.70|Positive|
|0.20|Negative|

Looks good.

---

### Model B

| Prediction | Reality |
|------------|---------|
|0.99|Positive|
|0.98|Positive|
|0.97|Positive|
|0.96|Negative|

Ranking is almost identical.

Classification accuracy may be identical.

But probabilities are terrible.

The model claimed

96%

yet failed.

That means it's overconfident.

---

A calibrated model satisfies

```
Among all customers predicted 80%

about 80%

actually become positive.
```

If

10,000 customers

receive

0.80

then roughly

8,000

should actually belong to the positive class.

That's calibration.

---

### Metrics here are different

Instead of precision

we use

- Log Loss
- Brier Score
- Calibration Curve
- Expected Calibration Error

Notice none of these require a threshold.

---

# Layer 3: Business Decision

Now suppose probabilities are perfectly calibrated.

What action should we take?

Entirely different question.

Suppose

```
P(churn)=0.40
```

Should we send a retention offer?

Maybe.

Depends on economics.

---

Suppose

Saving a customer earns

$500

Retention email costs

$2

Then contacting someone with

40%

chance

is profitable.

---

Now suppose

Offer costs

$150

Suddenly

40%

may no longer justify intervention.

The model is identical.

Only business changes.

---

Even more interesting.

Suppose call center capacity

```
1,000 calls/day
```

Model predicts

```
100,000 customers
```

Now there isn't even a threshold problem anymore.

Instead

Rank everyone

Take top 1,000

This becomes a ranking optimization problem.

---

# These Three Problems Are Independent

Imagine two models.

## Model A

Excellent ranking

Terrible probabilities.

```
Customer A 0.99
Customer B 0.95
Customer C 0.90
```

Actually

```
True risks

0.62
0.60
0.58
```

Ranking perfect.

Calibration awful.

---

## Model B

Excellent calibration

Slightly weaker ranking.

Which one wins?

Depends entirely on application.

---

### Marketing campaign

Need top customers.

Ranking matters.

Calibration matters much less.

---

### Insurance pricing

Need accurate probabilities.

Calibration becomes critical.

---

### Medical diagnosis

Need calibrated risk estimates

plus decision thresholds

plus physician costs

plus downstream interventions.

Entirely different optimization.

---

# Why Imbalanced Learning Makes This Confusing

Suppose fraud rate

```
0.5%
```

People often say

> Let's oversample.

Why?

Usually because

```
Recall improved.
```

But what actually happened?

Three possibilities.

---

## Classification improved

Maybe yes.

---

## Calibration worsened

Also possible.

Oversampling changes the effective class distribution during training, so the predicted probabilities may no longer reflect the real-world prevalence unless recalibrated.

---

## Business performance worsened

Possible too.

You doubled recall

but increased false positives

10×

Operations can't investigate all alerts.

Business loses money.

---

# A Real Example

Suppose

100,000 transactions

Fraud rate

1%

```
1,000 frauds
99,000 legitimate
```

---

## Model A

Threshold 0.5

```
Recall = 60%

Precision = 60%
```

Investigations

```
1000 cases
```

Operationally feasible.

---

## Model B

Threshold 0.2

```
Recall = 95%

Precision = 10%
```

Now investigations become

```
9,500 cases
```

The fraud team only has capacity for 1,000 investigations.

Model B has much better recall.

Yet it's unusable.

---

Junior conclusion

> Better recall!

Senior conclusion

> We optimized the wrong objective.

---

# The Senior Mental Model

Think of an ML system as three stacked layers:

```
Business Objective
────────────────────────
Which action should we take?
Threshold
Capacity
ROI
Costs

↑

Probability Layer
────────────────────────
Is 0.80 really 80%?

Calibration
Log Loss
Brier Score

↑

Ranking Layer
────────────────────────
Who is riskier than whom?

ROC-AUC
PR-AUC
Precision
Recall
F1
```

Each layer answers a different question:

- **Ranking layer:** *Can the model correctly order examples from most likely to least likely?*
- **Probability layer:** *Do the predicted probabilities correspond to real-world frequencies?*
- **Decision layer:** *Given costs, benefits, and operational constraints, what action should we take?*

A common mistake is to optimize only the ranking layer (for example, maximizing recall or F1) and assume the whole system is optimized. In production, success often depends just as much on probability calibration and business decision design as it does on the classifier itself.

This is why senior practitioners often say:

> **"We're not building a classifier—we're building a decision system."**

The classifier is only one component. The real value comes from combining accurate ranking, trustworthy probabilities, and decisions that align with business objectives and operational constraints.