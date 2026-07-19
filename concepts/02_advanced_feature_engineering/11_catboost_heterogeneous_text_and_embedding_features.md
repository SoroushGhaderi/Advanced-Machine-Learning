This is one of CatBoost's lesser-known but genuinely powerful capabilities. Most people think of CatBoost only as "the algorithm that handles categorical variables," but it is actually designed to work with **multiple data modalities** within a single gradient-boosting model.

Here's what this means from a senior data scientist's perspective.

---

# 8. Broader input types are optional

Most tree-based models expect every feature to already be numeric. If your dataset contains free text or vector embeddings, you usually need a completely separate preprocessing pipeline before training.

CatBoost can perform much of this preprocessing internally.

It supports four major feature types simultaneously:

- Numerical features
- Categorical features
- Text features
- Embedding features

Instead of manually converting every input into engineered numeric columns, CatBoost can transform these feature types into representations suitable for tree learning during training.

The important point is that **these are still tree models**—CatBoost is not replacing language models or neural networks. It is simply providing native preprocessing so these richer feature types can participate in gradient boosting.

---

## Numerical features

These are ordinary continuous variables.

Examples

- age
- salary
- balance
- account_age
- number_of_transactions

Nothing special happens here—they are handled like any gradient boosting library.

---

## Categorical features

These are CatBoost's primary strength.

Examples

- occupation
- education
- city
- product_category
- device_type

Rather than requiring one-hot encoding or manual target encoding, CatBoost learns leakage-safe ordered statistics automatically.

This is why CatBoost is often the first choice for structured business data.

---

## Text features

Suppose a banking dataset contains

| Customer | Complaint |
|-----------|-----------|
| A | "Mobile app crashes when paying bills." |
| B | "Excellent customer service." |
| C | "Loan application took too long." |

Traditional gradient boosting cannot use this column directly.

Normally the pipeline becomes

```
Text

↓

TF-IDF
or
Bag-of-Words
or
Sentence Embedding

↓

Numeric Matrix

↓

XGBoost
LightGBM
Random Forest
```

CatBoost allows a much simpler workflow.

```
Raw text

↓

CatBoost text processing

↓

Tree model
```

Internally it can create text-derived numeric features (using configurable text-processing pipelines) that are then used by the trees.

This removes a significant amount of feature-engineering code.

---

## Embedding features

Modern machine learning increasingly represents objects as vectors.

For example

Customer profile

```
[0.13,
-0.42,
0.88,
...
0.19]
```

or

Sentence embedding

```
768 dimensions
```

or

Product embedding

```
512 dimensions
```

Instead of expanding these into hundreds of independent columns and treating each dimension separately in your preprocessing code, CatBoost can accept fixed-length embedding vectors directly as a feature type.

This is especially useful when another model has already learned a meaningful representation.

Examples include

- BERT sentence embeddings
- OpenAI embeddings
- product embeddings
- image embeddings
- customer embeddings learned from historical behavior

CatBoost then learns tree splits using these vector representations alongside traditional structured features.

---

# Why this matters

Imagine a customer churn problem.

Instead of using only

```
Age
Balance
Products
Income
Tenure
```

you can also include

```
Last customer complaint
```

or

```
Support chat transcript
```

or

```
Email subject
```

or

```
Product description
```

within the same model.

Likewise, you might include a precomputed embedding that summarizes a customer's browsing history or purchase behavior.

The result is a single model that combines structured business variables with richer contextual information.

---

# But there are important boundaries

This capability is useful only when the additional modalities genuinely contain predictive information.

Adding text or embeddings simply because CatBoost supports them can increase complexity without improving performance.

A senior data scientist treats these features like any other feature family:

- define a hypothesis for why they should help,
- verify they are available at prediction time,
- compare against a structured-data baseline,
- evaluate them with proper out-of-fold experiments,
- retain them only if they deliver meaningful, reproducible improvements.

In many tabular business problems—such as credit scoring, insurance pricing, or customer segmentation—structured numerical and categorical variables already capture most of the predictive signal. In those cases, introducing text or embeddings may not justify the added engineering and operational cost.

---

## Senior takeaway

The key insight is not that CatBoost "supports text."

The important idea is that **CatBoost allows heterogeneous data sources to be modeled within a single gradient-boosting framework**. When your prediction problem naturally combines structured data with textual or embedding-based information, CatBoost can simplify the pipeline by handling these feature types natively. However, native support is a convenience—not a reason to include them. They should be added only when a disciplined, leakage-safe evaluation shows they provide measurable business value.
