This is one of the most important conceptual shifts from "building models" to **building decision systems**.

Many data scientists unintentionally mix **three different problems**:

1. Learning a score
2. Estimating a probability
3. Making a business decision

These are related, but they are **not the same optimization problem**.

---

# Layer 1 — Raw Scores (What the model actually learns)

Gradient boosting doesn't directly learn probabilities.

Instead, it learns a function

\[
F(x)
\]

called the **raw score**, **margin**, or **logit**.

Think of it as accumulated evidence.

Each tree simply says

> "Increase confidence a little."

or

> "Decrease confidence a little."

The ensemble becomes

\[
F(x)=F_0(x)+\eta h_1(x)+\eta h_2(x)+...
\]

Notice nothing here mentions probabilities.

Every tree simply nudges this raw score.

Example:

Initial prediction

\[
F_0=-1.2
\]

Tree 1

+0.35

Tree 2

−0.18

Tree 3

+0.42

Final

\[
F= -0.61
\]

The model is finished.

No probabilities yet.

---

# Layer 2 — Convert score into probability

Only after training do we convert

\[
F
\]

into

\[
p
\]

using the sigmoid.

\[
p=\frac1{1+e^{-F}}
\]

Example

Raw Score | Probability
-----------|------------
−4 | 0.018
−2 | 0.119
0 | 0.50
2 | 0.881
4 | 0.982

Notice something interesting.

A score of

4

doesn't mean

400%

confidence.

It simply maps through the sigmoid.

The model is really learning

> evidence

not

> probability.

The sigmoid interprets that evidence.

---

# Why don't we learn probabilities directly?

Because probabilities have annoying constraints.

They must stay between

0

and

1.

Raw scores can be anything.

\[
(-\infty,\infty)
\]

This makes optimization much easier.

Gradient descent works naturally in unconstrained space.

---

# Layer 3 — Decision Threshold

Now comes something completely different.

Suppose the model predicts

Customer | Probability
---------|------------
A | 0.91
B | 0.72
C | 0.61
D | 0.49
E | 0.31

The model stops here.

It never says

> "Call customer."

That decision belongs to you.

You choose

```text
Threshold = 0.5
```

Then

Customer | Action
---------|--------
0.91 | Call
0.72 | Call
0.61 | Call
0.49 | Don't call

But why 0.5?

There's nothing magical about it.

---

# The threshold is a business decision

Imagine:

Each phone call costs

$2

A successful conversion earns

$500

Suddenly,

calling someone with

40%

chance might be profitable.

Maybe your threshold should be

0.23

instead.

Another company might have

$50 acquisition cost.

Now the optimal threshold might become

0.81

Same model.

Same probabilities.

Different economics.

Different threshold.

Nothing about machine learning changed.

---

# This is why F1 depends on threshold

Suppose probabilities are

```text
0.91
0.84
0.72
0.64
0.58
0.49
0.42
```

Threshold = 0.5

Lots of positives.

High recall.

Lower precision.

Threshold = 0.8

Few positives.

Higher precision.

Lower recall.

The model never changed.

Only the decision rule changed.

That's why F1 changes.

---

# But Log Loss doesn't care about threshold

Suppose

Actual

```text
1
```

Model A predicts

```text
0.90
```

Model B predicts

```text
0.60
```

Both predict

Positive

at threshold

0.5.

Classification accuracy says

Both are correct.

Log loss says

Model A is much better because

90%

was closer to reality.

This is why log loss evaluates **probability quality**, not classification decisions.

---

# Calibration is another completely different problem

Imagine your model predicts

0.80

for

1,000 customers.

After one month,

800 actually convert.

Excellent.

The model is calibrated.

Now imagine

another model predicts

0.80

for

1,000 customers

but only

500 convert.

The ranking may still be excellent.

AUC may still be high.

Accuracy might still be high.

But the probabilities are wrong.

This matters enormously because many downstream systems rely on the probabilities themselves:

- Expected revenue calculations
- Inventory planning
- Risk management
- Medical risk estimation
- Insurance pricing
- Capacity planning

If the probabilities are systematically too high or too low, those decisions become biased.

---

# This leads to four distinct optimization problems

| Stage | Objective | Typical Metric | Who owns it? |
|--------|-----------|----------------|--------------|
| Learn the model | Fit the underlying function | Log loss, RMSE | ML algorithm |
| Produce reliable probabilities | Match predicted probabilities to observed frequencies | Calibration curve, Brier score, Expected Calibration Error (ECE) | ML engineer |
| Rank observations correctly | Put higher-risk cases above lower-risk ones | ROC AUC, PR AUC | ML engineer |
| Decide what action to take | Maximize business value | Profit, ROI, cost, utility | Product owner / business stakeholders |

This separation is easy to overlook, but it's fundamental.

A model can have an excellent AUC yet produce poorly calibrated probabilities. A perfectly calibrated model may still lead to poor business outcomes if the operating threshold is chosen badly. Conversely, a simple threshold adjustment can significantly improve business performance without changing the model at all.

## The senior mindset

Junior practitioners often ask:

> "What's the best threshold?"

A senior data scientist asks:

> "What decision are we trying to optimize, what are the costs and benefits of each outcome, are the probabilities trustworthy enough for that decision, and how should we choose the operating point given those constraints?"

That's why I often describe a production ML system as a pipeline with three distinct stages:

```text
Model
        │
        ▼
Raw Score (logit)
        │
        ▼
Probability
        │
        ▼
Calibration (if needed)
        │
        ▼
Business Threshold
        │
        ▼
Action
```

Each stage solves a different problem. Treating them independently—rather than assuming "a good model" automatically produces good decisions—is one of the biggest distinctions between building models for a notebook and building systems that create value in production.