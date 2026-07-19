This is one of the most important mindset shifts from junior to senior ML engineering.

> **Leakage is not "using a highly predictive feature." Leakage is using information that would not exist when the model is actually asked to make a prediction.**

Let's unpack why.

---

## The key question: *When is the prediction made?*

Every ML system has a **prediction moment**.

For the bank marketing dataset, suppose the business process is:

```
Today 9:00 AM
│
├── Decide who to call
│
├── Call customer
│
├── Conversation lasts 8 minutes
│
├── Customer says yes/no
│
└── Record call duration
```

The prediction happens **before the call**.

At that moment, the model can only use information that already exists.

---

## Why `duration` is leakage

The dataset contains

```
duration = 523 seconds
```

Looks harmless.

But ask yourself:

> How would the model know the call lasted 523 seconds before the call even happened?

It can't.

The feature is created **after** the prediction.

Graphically:

```
Prediction time
      │
      ▼

customer age
balance
housing loan
job
education
...
duration  ❌ not known yet
```

Using it is equivalent to letting the model peek into the future.

---

## Why does it improve accuracy so much?

Because call duration is strongly connected with the outcome.

Think about human behavior.

```
Call lasts 20 seconds

"Not interested."
```

Probably negative.

---

```
Call lasts 12 minutes

Customer asks questions
Talks about interest rates
Requests paperwork
```

Much more likely positive.

So the model learns

```
if duration > 500 sec
    probability ≈ 0.95
```

That rule is excellent...

...but completely useless before making the call.

---

## This is why leakage is dangerous

Imagine two models.

### Model A

Uses only valid features.

```
AUC = 0.82
```

---

### Model B

Uses `duration`.

```
AUC = 0.96
```

Many beginners think:

> "Great! We built a much better model."

A senior data scientist immediately thinks:

> "Impossible. What information leaked?"

An unexpectedly large performance jump often signals a flaw in the experimental setup rather than a genuine breakthrough.

---

## Leakage is about the timeline

Think of features as living on a timeline.

```
Past ---------------------- Future

Customer age
Income
Balance
Previous campaign
Employment

Prediction

Phone call

Duration
Conversation notes
Customer accepted offer
```

Everything to the **right** of prediction is forbidden.

---

## A better mental model

Instead of asking

> Is this feature correlated?

Ask

> **Could an API provide this value at prediction time?**

If the answer is "no, because it hasn't happened yet,"

then the feature is leaking future information.

This simple question catches a surprising amount of leakage in practice.

---

## Leakage is broader than future columns

Many people think leakage only means features like `duration`.

In reality, there are several common forms.

### 1. Future leakage

```
hospital discharge date
loan repayment status
next month's revenue
call duration
```

Information from the future.

---

### 2. Train-test contamination

```
StandardScaler.fit(all_data)
```

instead of

```
StandardScaler.fit(train_only)
```

The scaler has already "seen" the validation distribution.

---

### 3. Target leakage

Encoding categories using the entire dataset's target statistics.

For example,

```
city

New York -> 0.81
Boston -> 0.43
Chicago -> 0.15
```

If those averages were computed using validation rows, the model indirectly learns the validation labels. This is exactly why methods such as CatBoost's ordered target encoding were developed—they estimate target statistics without exposing each example to its own target or to future observations.

---

### 4. Duplicate/entity leakage

```
Customer A appears in training
Customer A appears in validation
```

The model isn't generalizing—it is recognizing the same entity.

---

## Availability is contextual

This is where senior thinking becomes important.

Consider the feature

```
previous
```

(number of previous contacts)

Is it valid?

The answer is:

> **It depends.**

If prediction occurs

```
10 minutes before today's call
```

then previous contacts already exist.

Valid.

---

If prediction occurs

```
before the marketing campaign even starts
```

those contacts haven't happened.

Invalid.

The exact same column can be valid or invalid depending on the operational workflow.

That's why experienced teams document not only a feature's meaning but also its **availability contract**—the point in the business process at which the feature is guaranteed to exist.

---

## A production mindset

When building a model, experienced ML engineers often work through a checklist like this:

| Question | Why it matters |
|----------|----------------|
| When is the prediction made? | Defines the decision point. |
| Which features exist before that moment? | Prevents future leakage. |
| Could this value be retrieved by the serving system? | Ensures deployability. |
| Is preprocessing fitted only on training data? | Prevents train/validation contamination. |
| Will this feature still exist six months from now? | Guards against brittle pipelines. |
| Is this feature generated by the target itself? | Prevents target leakage. |

## The principle to remember

One of the best heuristics in production ML is:

> **A model should never have access to information that the decision-maker wouldn't have at the moment the decision is made.**

If a loan officer, doctor, fraud analyst, or marketing system cannot know a piece of information when making the decision, then neither should the model. Framing leakage in terms of the **decision timeline** rather than just the dataset is what separates robust, deployable ML systems from models that only perform well in offline experiments.