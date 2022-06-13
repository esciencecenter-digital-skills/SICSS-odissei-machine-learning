---
title: Best Practices on Conducting ML Projects
---

# Before starting a machine learning project
## Ask yourself

:::incremental
- What is your scientific problem?
- Can this scientific problem be transformed to machine learning problem?
  - Keep the problem simple; if not, decompose it
- Do you really have to use machine learning?
:::

## Continue asking...

:::incremental
- What is the goal of your ML project?
- Do you have enough high-quality data?
- How do you measure the model performance?
  - First design and implement metrics
- Do you have good enough infrastructure?
:::

# During doing machine learning
## Pipeline or workflow
- Having a bad pipeline is better than nothing
  - Make one and optimize it

## Data
:::incremental
- Be patient with data engineering
- Split data to training, evaluation and test datasets
- Use common-sense features
- Borrow features from state-of-art models
:::


## Model
:::incremental
- Keep your fist model simple
  - Make it work is half of the success
- Be more patient with training
  - It is an iterative cycle to improve your model
- "Measure first, optimize second", design metrics for specific issues
:::

# After training

## Versioning
- Version your data, code and everything

## Re-train
- Retrain the model when possible
  - e.g. more new data
