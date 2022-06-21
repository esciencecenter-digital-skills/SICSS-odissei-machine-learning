---
title: Performance metrics
parallaxBackgroundImage: image/e-content1.png
title-slide-attributes:
    data-background-image: image/e-title.png
    data-background-size: cover
---

# Confusion matrix

![](image/2.1-Confusion-matrix.png)

Note that the majority class is "No BBQ".

::: notes
Suppose we have this confusion matrix.

Note that the majority class is "No BBQ".

In machine learning jargon, when we have binary classification we speak about 'positive' and 'negative' classes.
It is a bit arbitrary which class you choose as positive, but generally speaking it's the class you're most interested in to predict well. We are interested in nice BBQ weather, so this is our 'positive' class.
:::


# TP TN FP FN

::: fragment
![](image/2.2-Confusion-matrix-rates.png){#id .class height=250px}

- TN: True Negative
- FN: False Negative
- FP: False Positive
- TP: True Positive
:::

::: notes
- **T**rue **P**ositives: Positive in reality, prediction positive
- **T**rue **N**egatives: Negative in reality, prediction negative
- **F**alse **P**ositives: Negative in reality, prediction positive
- **F**alse **N**egatives: Positive in reality, prediction negative
:::

# Accuracy

::: fragment
*Accuracy* is the rate of correct predictions:
$$acc = (TP+TN) / (TP+TN+FN+FP)$$

In our case:
$$acc = (70 + 15) / 100 = 0.85$$

Note that this score gets skewed by the majority class!
:::

# Precision & recall

::: fragment
From `Positive` perspective:

- Precision: How many of the predicted BBQ days can we truly fire our BBQ?
- Recall: How many of the true BBQ days were predicted by the model?
:::

::: fragment
$$precision = TP / (TP + FP) = 15 / (15+10) = 0.6$$
$$recall = TP / (TP + FN) = 15 / (15+5) = 0.75$$
:::

::: notes
Note how these scores are lower than the precision. The model is not doing so well on our class of interest.
:::

# f1

::: fragment
Often you want both precision and recall to be high.

We can calculate the f1 score:
$$f1 = 2pr / (p + r)$$
where p is precision and r is recall
:::

::: notes
(Note: this is a 'harmonic mean', which gives more weight to low values compared to regular mean. so it's only high when both values are high).
:::

# Trade-off Precision/recall

---

For most models, we do not just get the predicted class, but also an associated *score*.

![](image/2.3-scores-table.png)

If the score is above a certain threshold, it assigns that class.

---

## Exercise

What do you think happens with the precision and recall when we increase or decrease the threshold?

![](image/2.3-scores-table.png)

---

## Solution

- **increase** the threshold, we get more strict: recall drops, precision may improve (if our model does well).
- **decrease** the threshold, recall may increase but precision could drop.

![](image/2.4-precision-recall-graph.png)

# Thank you {background-image="image/e-end1.png"}
## Q&A {background-image="image/e-end1.png"}