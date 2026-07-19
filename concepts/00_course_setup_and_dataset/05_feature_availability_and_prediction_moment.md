This is one of the most important mindset shifts from junior to senior ML engineering:

> **A feature is not "valid" because it exists in the dataset. It is valid only if it exists at the moment the prediction is made.**

Many people think about leakage as a property of a column. Senior data scientists think about leakage as a property of the **entire prediction workflow**.

## Think in terms of a timeline

Imagine a marketing campaign.

```
Customer history
       │
       ▼
──────────────────────────────────────────────────────────────► time

1 week before campaign      Call scheduled        Call starts      Call ends

       ↑                         ↑                   ↑               ↑
   Model could score        Another scoring      Duration grows   Outcome known
```

Every feature has a timestamp.

The model can only use information that already exists **before the prediction moment**.

The prediction moment is sometimes called the **decision point**.

---

## The same feature can be valid or invalid

This surprises many people.

Consider the feature:

```
campaign
```

It represents the number of contacts made during the current campaign.

Is it valid?

The answer is:

**It depends.**

### Scenario 1 — Score customers before the campaign begins

```
Campaign hasn't started yet.

campaign = ?
```

The value doesn't exist.

Using it would be impossible.

Therefore

```
campaign ❌ invalid
```

---

### Scenario 2 — Score immediately before today's phone call

Suppose the campaign has already contacted this customer twice.

```
campaign = 2
```

That information already exists.

Now

```
campaign ✅ valid
```

Nothing magical changed.

Only the **prediction moment** changed.

---

## Another example: previous

```
previous
```

Number of contacts before this campaign.

If all previous campaigns have already happened,

```
previous = 4
```

is perfectly legitimate.

But imagine you train one model in January and deploy it in October.

During those months,

```
previous
```

will continue changing.

If your deployment assumes a frozen value while production receives updated values, you've changed the data distribution.

---

## pdays

```
pdays
```

Days since last contact.

Again...

Imagine scoring customers today.

```
Last contacted 37 days ago

pdays = 37
```

Perfectly valid.

Now imagine scoring one week before today.

```
Last contacted 30 days ago

pdays = 30
```

Different value.

Nothing is wrong.

The feature simply depends on **when inference happens**.

---

## day and month

People often assume calendar variables are always available.

Not necessarily.

Suppose you generate a call list every Monday.

At that point you don't know

```
day = 24
```

because the exact call date hasn't been assigned yet.

If instead you're scoring customers five minutes before dialing,

then

```
day
month
```

are known.

Again:

Same dataset.

Same model.

Different operational workflow.

Different valid features.

---

## The famous duration example

This one is always invalid for pre-call prediction.

```
Call starts

↓

Conversation lasts 430 seconds

↓

duration = 430
```

When predicting whether to call,

the conversation hasn't happened yet.

So

```
duration
```

comes from the future.

It creates massive leakage.

The model effectively learns

> "Long conversations often end in subscription."

Which is true.

But completely useless before dialing.

---

## The real lesson

A feature should never simply be documented as

```
Available: Yes
```

Instead, senior ML teams document

| Feature | Available when? |
|----------|-----------------|
| age | Always |
| balance | Daily snapshot before scoring |
| previous | At campaign start |
| campaign | After first campaign contact |
| duration | After call ends (never for pre-call prediction) |
| poutcome | After previous campaign closes |

Notice how every feature has an associated **availability condition**.

---

## This becomes a Feature Contract

A mature feature contract looks something like:

```
Feature: campaign

Definition:
Number of contacts during current campaign.

Availability:
Only after the customer has already been contacted
during the current campaign.

Refresh:
Real-time after each completed call.

Allowed for models:
✓ Retargeting model
✗ First-contact propensity model
```

Notice that "allowed for models" depends on the business use case.

---

## Why senior data scientists ask different questions

A junior engineer often asks:

> Is this feature predictive?

A senior engineer asks:

> **When exactly does this feature become available?**

Because a feature with enormous predictive power is worthless if it arrives **after** the decision has already been made.

This is why experienced ML practitioners spend so much time defining the **prediction moment** before writing any modeling code. Once that moment is fixed, the timeline determines which features are legal. Leakage, validation strategy, feature engineering, and even deployment architecture all follow naturally from that single design decision.