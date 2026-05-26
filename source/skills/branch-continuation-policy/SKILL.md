---
name: branch-continuation-policy
description: Decide whether a user question should continue an active analytical branch, create/switch branches, or use global reasoning.
---

# Branch Continuation Policy

Use this skill before binding a new user question to prior branch context.

Continue an existing branch only when the user clearly depends on the prior result:
- remove outliers or exclude extreme records;
- compare against a named segment or previous result;
- explain this chart or this table;
- ask which groups remain leaders;
- show median instead or another transformation of the same metric/dimension;
- explicitly references the previous artifact, chart, table, or result.

Create or switch to a separate branch when the user asks for:
- a new metric, dimension, dataset, or chart family;
- a new business, customer-behavior, anomaly, strategic, or data-quality question;
- a new multi-dataset relationship, joinability, missing-links, or warehouse-design question.

Use global reasoning for:
- business risks, executive summaries, strategic recommendations, and “what can we investigate?”;
- cross-dataset questions, joinability, missing links, and warehouse design.

Escape analytical branch continuation entirely for non-analytical intent:
- creative or metaphorical comparisons, including movies, books, songs, jokes, or cultural references;
- casual conversation;
- project/meta questions about how the agent works, why an answer looked wrong, or what to test next;
- harmless out-of-scope questions.

For those questions, answer conversationally or ask a short clarification. Do not run grouped dataframe analysis, metric resolution, or active-branch fallback.

Never answer a new strategic, cross-dataset, creative, or conversational question with an old grouped-analysis branch. Do not say “the active comparison is still,” and do not expose branch, planner, executor, or fallback state.
