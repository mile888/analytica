---
name: multi-dataset-reasoning
description: Answer cross-dataset questions with semantic, joinability, missing-link, and warehouse-design reasoning without hallucinating joins.
---

# Multi-Dataset Reasoning

Use this skill for questions comparing, relating, joining, or designing around multiple datasets.

Select one answer mode:
- Schema comparison: compare row counts, column counts, exact shared columns, and dataset purpose.
- Semantic relationship: explain dataset purposes, shared business themes, compatible concepts, and limits.
- Joinability: identify stable shared keys, distinguish weak conceptual bridges from reliable identifiers, and state whether row-level joins are safe.
- Missing links: name important questions that cannot be answered and the identifiers or fields required.
- Warehouse design: propose shared entities, fact tables, dimension tables, bridge tables, and required keys.

Do not treat “no shared columns” as the whole answer. Lack of identical columns does not mean no analytical relationship exists.

Never hallucinate joins, correlations, shared keys, or exact semantic equivalence. If fields are only conceptually related, say so and keep the answer at aggregate or design level.

Do not collapse cross-dataset questions into the active single-dataset branch.
