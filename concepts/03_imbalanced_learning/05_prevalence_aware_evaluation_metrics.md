This is one of the most misunderstood topics in imbalanced learning. The phrase **"metrics must be prevalence-aware"** is much deeper than it first appears.

The central idea is:

> **Some metrics measure properties of the classifier. Others measure properties of the environment in which the classifier operates.**

The prevalence (also called the **base rate** or **class prior**) belongs to the environment.

---

# What is prevalence?

Prevalence is simply

\[
P(Y=1)
\]

the proportion of positive cases in the real population.

For example:

- Fraud detection → 0.1%
- Disease screening → 2%
- Customer churn → 15%
- Click prediction → 0.01%

The model does **not** determine prevalence.

The world does.

---

# Why prevalence matters

Imagine you build a fraud detector.

Training data:

| Legit | Fraud |
|--------|--------|
|500|500|

You intentionally balanced the dataset.

Your classifier predicts

- TP = 450
- FP = 50

Precision

\[
\frac{450}{450+50}=90\%
\]

Looks fantastic.

---

Now deploy it.

Real production prevalence:

| Legit | Fraud |
|--------|--------|
|999,000|1,000|

Suppose

Sensitivity = 90%

Specificity = 90%

Now:

Frauds detected

\[
900
\]

False positives

10% of 999,000

\[
99,900
\]

Precision becomes

\[
\frac{900}{900+99,900}
=0.89\%
\]

Your model went from

> **90% precision**

to

> **0.9% precision**

without changing a single parameter.

Only the prevalence changed.

---

# This surprises many people

People often think

> Precision measures the model.

It doesn't.

Precision measures

> **the quality of the model operating inside a particular population.**

Change the population.

Precision changes.

---

# Mathematical explanation

Precision is

\[
P(Y=1|\hat Y=1)
\]

Using Bayes' theorem,

\[
P(Y=1|\hat Y=1)
=
\frac{
P(\hat Y=1|Y=1)P(Y=1)
}
{
P(\hat Y=1)
}
\]

Expand denominator

\[
=
\frac{
\text{Recall}\times\text{Prevalence}
}
{
\text{Recall}\times\text{Prevalence}
+
(1-\text{Specificity})(1-\text{Prevalence})
}
\]

Notice something important.

Prevalence appears explicitly.

That means

**Precision is mathematically dependent on prevalence.**

No escaping it.

---

# Recall does NOT depend on prevalence

Recall is

\[
P(\hat Y=1|Y=1)
\]

Conditioning on actual positives.

Changing how many negatives exist changes nothing.

If there are

100 positives

or

10 million negatives

recall is still

\[
\frac{TP}{TP+FN}
\]

No prevalence term.

---

# Specificity also ignores prevalence

\[
\frac{TN}{TN+FP}
\]

Only negatives matter.

Changing positive frequency changes nothing.

---

# ROC-AUC is prevalence-independent

ROC uses

TPR

versus

FPR

Neither contains prevalence.

Therefore

ROC-AUC remains almost unchanged when prevalence changes.

This is why ROC-AUC is excellent for measuring ranking ability.

---

# PR-AUC depends on prevalence

Precision changes.

Therefore PR curve changes.

Therefore PR-AUC changes.

This is why PR-AUC is often more realistic for highly imbalanced problems.

---

# Why balancing the dataset can fool you

Suppose you oversample.

Original

99:1

After SMOTE

50:50

You evaluate precision on the balanced validation set.

You might observe

Precision = 94%

Looks incredible.

But production is still

99:1.

Actual precision might be

15%

Nothing "went wrong."

You evaluated in the wrong population.

---

# Senior data scientists distinguish two distributions

Training distribution

Used for optimization.

May be balanced.

May use SMOTE.

May use weighting.

Production distribution

Real world.

Must remain untouched.

Metrics reported to stakeholders should reflect production.

Not artificial training distributions.

---

# Balanced accuracy

Balanced accuracy is

\[
\frac{\text{Sensitivity}+\text{Specificity}}{2}
\]

Notice what's missing.

No prevalence.

It gives equal weight to

positive class

negative class

even if

99.99%

of observations belong to one class.

---

Suppose

Recall

95%

Specificity

80%

Balanced accuracy

\[
\frac{95+80}{2}
=
87.5\%
\]

Sounds excellent.

---

But imagine

Only

0.1%

of customers churn.

Suppose contacting one customer costs \$5.

The model now generates

hundreds of thousands of false alarms.

Balanced accuracy never notices.

Because it intentionally ignores prevalence.

---

# Why balanced accuracy exists

It solves a different problem.

Accuracy

\[
\frac{TP+TN}{N}
\]

can become meaningless.

Example

99.9%

negative

Always predict negative.

Accuracy

99.9%

Wonderful?

No.

Balanced accuracy fixes this by equally rewarding performance on both classes.

It asks

> Can the classifier discriminate both classes?

Not

> Does this make money?

---

# Business metrics care about prevalence

Imagine

Disease prevalence

0.01%

Hospital capacity

50 tests/day

False positives cost

\$100

False negatives cost

\$50,000

Balanced accuracy ignores all of this.

Business profit depends on

- prevalence
- intervention cost
- downstream treatment
- hospital capacity
- patient harm
- legal risk

None appear in balanced accuracy.

---

# Senior thinking

A junior practitioner might say:

> "Model A has higher balanced accuracy, so it's better."

A senior practitioner asks:

1. What is the production prevalence?
2. Is validation prevalence realistic?
3. Will prevalence drift over time?
4. Are we optimizing ranking or decisions?
5. What are the intervention costs?
6. What capacity constraints exist?
7. What happens if prevalence doubles?
8. Should thresholds change seasonally?

The metric is only meaningful in the context of the deployment environment.

---

# A useful mental model

Think of the metrics in three layers:

| Layer | Metric examples | Depends on prevalence? | Purpose |
|-------|-----------------|------------------------|---------|
| **Ranking** | ROC-AUC, C-index | No | Can the model order examples correctly? |
| **Classification** | Recall, Specificity, Balanced Accuracy | Mostly no | How well does a chosen threshold separate classes? |
| **Deployment** | Precision, PR-AUC, Expected Profit, Cost, Lift, ROI | Yes | How well does the system perform in the real population? |

This is why experienced ML practitioners often evaluate a model in stages:

1. **Ranking:** Does the model discriminate positives from negatives? (ROC-AUC, PR-AUC)
2. **Calibration:** Do predicted probabilities correspond to real event rates? (Brier score, calibration curves)
3. **Business policy:** Given the real production prevalence, costs, and operational constraints, what threshold or top-\(K\) strategy maximizes value?

The key insight is that **prevalence is a property of the deployment environment, not the model**. A model can have identical ROC-AUC, recall, and specificity in two different populations, yet produce dramatically different precision, profit, and operational workload because the underlying event rate has changed. That's why metrics used for deployment decisions must always be interpreted in the context of the real-world prevalence.