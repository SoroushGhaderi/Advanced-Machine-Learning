# Local visual assets

Visual assets are optional teaching material and are intentionally excluded
from Git. Keep them on the local filesystem under the module that uses them:

```text
assets/
├── 00_course_setup_and_dataset/
├── 01_gradient_boosting_fundamentals/
├── 02_advanced_feature_engineering/
└── 03_imbalanced_learning/
```

Use `snake_case` names that mirror the matching concept or notebook topic,
with the lesson number first when a visual belongs to a numbered lesson. The
notebook builders do not require these files, so a fresh clone remains
portable without downloading or committing images.
