---
title: Best Practices
parallaxBackgroundImage: image/e-content1.png
title-slide-attributes:
    data-background-image: image/e-title.png
    data-background-size: cover
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
- Are there any risks related to privacy and ethics?
  - [Deon ethics checklist](https://deon.drivendata.org/)
:::

::: notes
Biased data leads to biased model.

Suggestions about privacy & ethics risks:

- increase awareness of privacy and ethics
- discuss the risks before conducting project with stakeholder
- be careful with the limitations of others' ML model
- state the limitations of your model when make it public

[A famous paper about biased ML](https://proceedings.neurips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf)
:::

# During doing machine learning
## Workflow or pipeline
- Having a bad workflow is better than nothing
  - Make one and then optimize it

---

![](image/4.1-ML-project-time-costs.png){height=450px}
<!-- https://www.cloudfactory.com/data-labeling-guide -->

## Data

::: incremental
- Be very patient with data engineering
- Split data to training, validation and test sets
- NEVER mix using data:
  - training data only for training
  - validation data only for validation (picking model)
  - test data only for test (estimating generalization performance)
- Use common-sense features
- Borrow features from state-of-the-art models
:::


## Model
:::incremental
- Set a baseline performance/model
  - use state-of-the-art model
  - human performance
  - guess it with your experience
- Keep your first model simple
- Be patient with training
  - It is an iterative cycle to improve your model
:::


# After training

## Versioning
- Version your data, code and everything
  - using [git and github](https://github.com/)
  - [MLFlow](https://mlflow.org/)
  - [Weights & Biases](https://wandb.ai)

## Re-train
- Retrain the model when possible
  - e.g. new data, new features

# Thank you {background-image="image/e-end1.png"}
## Q&A {background-image="image/e-end1.png"}