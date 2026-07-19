This is arguably **the single most important mindset shift** between a junior and a senior data scientist.

Most feature engineering mistakes are not about mathematics—they're about **violating the prediction-time data contract**.

Let's unpack each idea.

---

# The Prediction-Time Data Contract

Every production model has an invisible contract:

> **The model may only use information that will actually exist when the prediction is made.**

Everything else is leakage.

Notice this definition has nothing to do with train/test splits.

It is about **time**.

Think of the prediction moment as a wall.

```
Past -----------------|---------------- Future
                      ^
              Prediction happens here
```

Everything on the left is legal.

Everything on the right is illegal.

---

# Example: Bank Marketing

Suppose a bank wants to predict

> "Should we call this customer today?"

The prediction occurs **before the phone call.**

Available information:

```
Age
Income
Balance
Previous campaign
Previous outcome
Days since last contact
```

Unavailable:

```
Call duration
Conversation notes
Customer reaction
Whether they eventually subscribed
```

Those literally don't exist yet.

---

# Why `duration` is leakage

Imagine the data contains

```
duration = 950 seconds
```

What does this tell us?

A customer that hangs on the phone for 15 minutes is much more likely to subscribe.

The model quickly learns

```
long duration
        ↓
customer probably says yes
```

The AUC jumps dramatically.

Looks amazing.

Except...

Can you know call duration **before making the call?**

No.

You would need to travel into the future.

This feature is essentially giving the model tomorrow's newspaper.

---

Imagine deployment.

Training:

```
Customer
↓

Call duration = 800 seconds
↓

Model predicts YES
```

Production:

```
Customer
↓

???

Call hasn't happened.

Duration doesn't exist.
```

Your best feature disappears.

The production model immediately collapses.

---

# Leakage isn't cheating intentionally

This is an important lesson.

Most leakage is accidental.

People think

> "It's in the dataset."

But datasets contain **historical facts**, not necessarily **available facts**.

Historical datasets often include

- future outcomes
- future behavior
- post-event information

because analysts collected everything after the event ended.

Production cannot.

---

# Senior question #1

Instead of asking

> "Does this improve AUC?"

A senior asks

> **"Will this exist at prediction time?"**

That question prevents most leakage.

---

# The hidden leakage in preprocessing

Leakage also happens inside preprocessing.

Example:

Suppose you have

```
Training

20
30
40
50
```

Validation

```
100
```

Now imagine StandardScaler is fit on

```
20
30
40
50
100
```

instead of only

```
20
30
40
50
```

Now the scaler has learned

- global mean
- global variance

using validation information.

Even though labels weren't used.

This is still leakage.

The validation distribution influenced training.

---

# Why this matters

Suppose the validation set contains

```
very wealthy customers
```

Now scaling shifts because of them.

The model indirectly adapts itself to validation.

Performance looks slightly better.

In production?

Those statistics won't exist.

---

# This applies to **every learned transformation**

Many people only think target encoding leaks.

Actually **anything that learns from data** can leak.

Examples:

### Imputation

Wrong

```
median(age)

computed on

train + validation
```

Correct

```
median(age)

computed only on train
```

---

### Scaling

Wrong

```
StandardScaler.fit(all_data)
```

Correct

```
StandardScaler.fit(train_only)
```

---

### Quantile bins

Wrong

```
Bin edges learned
using all rows
```

Correct

```
Bin edges learned
inside training fold
```

---

### Rare category grouping

Suppose

```
Paris
London
Tokyo
SmallTown
Village
```

You decide

```
< 20 observations

↓

OTHER
```

Question:

How do you know there are fewer than 20?

If you counted using

```
train + validation
```

you leaked.

Counts themselves are learned information.

---

### Target Encoding

Even more dangerous.

Suppose

```
Category A

targets

1
0
1
1
```

Average

```
0.75
```

If a row contributes to its own average

then

its own target influences its feature.

The model indirectly sees the answer.

Huge leakage.

---

# Why CatBoost is clever

CatBoost asks

For this row

```
Row #80
```

Pretend rows

```
81
82
83
...
```

haven't happened yet.

Only compute category statistics from

```
1
...
79
```

Now the current target cannot leak into itself.

This mimics real life.

---

# The subtle case: `pdays = -1`

This is one of my favorite examples because it shows why **semantics matter more than numbers**.

Dataset:

```
pdays

10
50
7
-1
40
```

Many beginners think

```
-1

means

minus one day
```

It doesn't.

It means

> **Customer has never been contacted before.**

That is not a number.

It is a completely different state.

Think of it like

```
Temperature

20°C
25°C
18°C
banana
22°C
```

Would you average "banana"?

Of course not.

Yet many algorithms effectively treat `-1` as if it were "less than zero days."

That creates meaningless relationships.

---

# Better representation

Instead of

```
pdays
```

create two features.

Feature 1

```
previously_contacted

0
1
```

Feature 2

```
days_since_previous_contact

NaN
10
35
120
```

Now the model learns separately

- whether contact ever happened
- how long ago it happened

These are distinct business concepts.

This representation is much more faithful.

---

# Senior thinking

A senior engineer doesn't view features as columns.

They view them as **measurements of the real world**.

Every feature should answer three questions:

### 1. Is it available?

```
Prediction time?
```

If not,

throw it away.

---

### 2. What does it actually measure?

Not

```
-1
```

but

```
Never contacted
```

Not

```
campaign = 7
```

but

```
Seventh attempt during this campaign
```

The semantics matter more than the raw values.

---

### 3. How was it learned?

Did preprocessing learn from

```
train only
```

or

```
all data
```

If the latter,

you have contamination.

---

## A mental model

Whenever you engineer a feature or apply a transformation, imagine you're standing at the exact moment a production prediction is made and ask:

> **"If I freeze time right now, could I compute this feature using only the information already available?"**

If the answer is **yes**, the feature satisfies the prediction-time data contract.

If the answer is **no**, you've introduced leakage—even if your cross-validation score improves. Those improvements are usually an illusion that disappears in production.

This way of thinking naturally guides decisions about feature engineering, preprocessing, validation, and even model evaluation, making it one of the foundational habits that distinguishes production-ready machine learning from notebook experimentation.
