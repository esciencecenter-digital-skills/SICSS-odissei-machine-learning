# Collaborative document day 2

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------


This is the Document for today: [link](https://tinyurl.com/mspb2kfk)

Collaborative Document day 1: [link](https://tinyurl.com/274bduec)

Collaborative Document day 2: [link](t)

# https://tinyurl.com/mspb2kfk

## 👮Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question during the lecture, please raise your hand :raising_hand: :man-raising-hand: 

For getting help during the exercises, we use a post-it system:
- :blue_heart: If you *need help*, please put the *blue post-it* on top of your laptop, and one of the helpers will come to help you
- :yellow_heart:  When you are *doing fine* with the exercises, please have the *yellow post-it* on your laptop




General tips and tricks:
- In JupyterLab/Jupyter Notebook, you can use `Shift+Tab` in a cell to get help on a function.

## 🖥 Links

Download files:
[Weather dataset](https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1)
[BBQ weather labels](https://zenodo.org/record/5071376/files/weather_prediction_bbq_labels.csv?download=1)

## 👩‍🏫👩‍💻🎓 Instructors

Djura, Cunliang, Dafne, Sven 

## 🧑‍🙋 Helpers

Kody, Christiaan, Laura



## 🗓️ Agenda
| Time | Topic |
|--:|:---|
| 9:00 - 9:30 | Welcome and recap of yesterday |
| 9:30 - 10:00 | Performance metrics |
| 10:00 - 10:45 | Cross validation |
| 10:45 - 11:00 | :coffee::tea: Break |
| 11:00 -  11:30 | Cross validation excercise  |
| 11:30 - 12:00 | Theory on regression  |
| 12:00 - 12:30 | Hands-on regression |
| 12:30 - 13:30 | :sandwich: Lunch break |
| 13:30 - 14:45 | Hands-on Regression and best practices |
| 14:45 - 15:00 | :coffee::tea: Coffee break |
| 15:00 - 16:00 | Liss data |
| 16:00 - 17:00 | lecture Joris Mulder |



## 🔧 Exercises

Discuss with your neighbour: how do you arrive at a threshold for classification?
What do you think happens with the precision and recall when we increase or decrease the threshold? Put your answer below.

Answers: 
- Prec up (few false positives since you are very certain before you code positive), rec down (many instances will not be picked up due to your high threshold)
- If you put a higher threshold precision goes up, recall goes down
- High threshold = Lower recall = High precision
- Higher threshold means slightly higher precision but lower recall. 
- If the threshold increases, FP will be reduced, so precision will increase 

### Exercise 1: Cross validation
- Why do we not want to use the test set for parameter optimization?
- What are advantages or disadvantages of cross validation over a single train-validation split?

Answers:
- If we use the test-set then we are fitting our model to the test set in a way
- Advantage is that you use all of the data available instead of splitting again, and you are less sensitive to random splits being in a specific direction. Downside is computational costs (that would be especially the case when K approaches N)

Answers: 
- Overfitting. 
- Making sure the model works properly before we apply it on the test data.

### Exercise 2: optimize different parameters
1. Look at the sklearn documentation on Random Forests, and choose one or more hyperparameters to tune.  Create a pipeline, looping over different parameters. What do you find? Can you improve over the best model so far?

2. Create visualizations to understand the relationship between the parameter values and the model performance. What do you learn from the plots?

3. Apply your best model to the test set. Are you confident that the model works well on new data?

 

### Regression: Exercise 1
Before you can apply machine learning, you will first have to identify the inputs and outputs to the algorithm. What will be the inputs in this case, and what will be the output?

Does our dataset have everything need or do we need to apply additional preprocessing?
### Regression: Exercise 2
Any ideas what could serve as a baseline performance?


### Regression: Exercise 3
Check the [scikit learn documentation](https://scikit-learn.org/stable/supervised_learning.html) for other regression models.

Pick an algorithm and swap out your MLPRegressor for that one.

- How does it perform on the training and test set?
- Is it overfitting, or not?
- Read the documentation on your chosen algorithm. Can you gain an intuition why this model performs the way it does?

Some suggestions:
- [Nearest neighbor regression](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor)
- [Random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor)

Tip: You can automate some parts of your model evaluation:

```python=
za
```


```python=
def evaluate_model(regressor):
    pipeline = Pipeline([('scaler', MinMaxScaler()), ('regressor', regressor)])    
    pipeline.fit(X_train, y_train)
    
    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    display('Training')
    plot_against_baseline(y_train, pred_train, baseline_train, axis=axes[0], label='Performance on training set')
    display('Test')
    plot_against_baseline(y_test, pred_test, baseline_test, axis=axes[1], label='Performance on test set')

```

## 🧠 Collaborative Notes

### (Classification) performance metrics:

Terminology:

[Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
False negatives (bottom left of matrix): instances that were predicted negative but incorrectly
True negatives (top left of matrix): instances that were predicted negative correctly
False positives (top right of matrix): instances that were predicted positive incorrectly
True positives (bottom right of matrix): instances that were predicted positive correctly

Calculation of accuracy:
$$(True Positives + True Negatives) / Total Instances$$

Calculation of precision:
$$True Positives / (True Positives + False Positives)$$

Calculation of recall:
$$True Positives / (True Positives + False Negatives)$$

Calculation of F1 score:
$$2 \cdot precision \cdot recall / (precision + recall)$$
The range of F1 score between 0 and 1.
- When precision or recall is 0, then F1 is 0;
- when precision and recall are perfect (there is no false positives or false negatives), then F1 is 1.

### Thresholds for classification:
What is threshold?
- the direct output of classifier is a probability score, and you need a threshold to classify the sample with that output score to a specific class.

Exercise: What do you think happens with the precision and recall when we increase or decrease the threshold?

Answer: When threshold increases, you are more strict in accepting instances of a class. Therefore, precision may either increase or stay the same (it does not necessarily increase because most models are not perfect). Recall conversely can tend to decrease because you permit or accept less positive classifications.

:bulb: **Tip**: You can plot precision and recall vs. classification threshold to observe the relationships between them, use [sklearn.metrics.precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html) to get the curve of the precision and recall tradeoff

### Cross validation:

What is [cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))?

- We divide / split up training data into parts. Depending on the data size you can choose how many parts. We will choose three. Then we iteratively train three models. Each part of the data gets a turn to be the train / test sets. Basically we are training and testing on different parts of the dataset and we compare the results to check how robust our model will be in practice. When the results of cross validation are not satisfactory, we can try to tune the hyperparameters of the model to improve its accuracy.

Disadvantages of cross validation:

+ Takes more time and resources to train our models (especially if you use many splits e.g. 10 or more)
+ ...
+ ...

**Tip**: decision about whether to uses cross validation or not: know your data, it depends a lot on the content of your data. If you use time-series data for example it can be tricky to split data because time is a factor and can introduce biases in the model based on the time periods chosen for test / train / validation sets.

:bulb: [A guideline for choosing split ratio](https://stackoverflow.com/questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio)

Applying cross validation:

Import and prepare the data:

```python=
import pandas as pd
weather_train = pd.read_csv('data/weather_train.csv')
features = weather_train.columns[:-1]
x = weather_train[features]
y = weather_train['BASEL_BBQ_weather']
```
Import sci-kit learn libraries that we will need:

```python=
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
```

Create our training pipeline for the data:

```python=
pipe = Pipeline([
    ('scale', MinMaxScaler()),
    ('model', RandomForestClassifier(max_leaf_nodes=3, random_state=0))
])
pipe.get_params()
```

Perform cross validation:

```python=
model_cv = GridSearchCV(estimator = pipe,
                                        cv = 3,
                                        param_grid = {
                                        'model__n_estimators' : [10, 50, 100, 250, 500]
                                        },
                                        scoring = "f1")
model_cv.fit(x, y)
```

View results of cross validation:

```python=
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results
```

Plot cross validation results:

```python=
import matplotlib.pyplot as plt
plt.scatter(cv_results['param_model__n_estimators'], cv_results['mean_test_score'])
plt.errorbar(cv_results['param_model__n_estimators'], cv_results['mean_test_score'], yerr = cv_results['std_test_score'])
```

### Regression:

Applying regression in Python:


Import libraries:

```python=
import seaborn as sns
import pandas as pd
import numpy as np
```

Import data:

```python=
data = pd.read_csv('https://zenodo.org/record/5071376/files/weather_prediction_dataset.csv?download=1')
data.head()
```

Prepare data for regression:

```python=
n_rows = 365*3
weather_3years = data[:n_rows].drop(columns=['DATE', 'MONTH'])
sunshine_tomorrow = data[1:n_rows + 1]['BASEL_sunshine'].values
sunshine_tomorrow.mean() # see the mean value of sunshine hours in Basel
```

Split data into train / test sets:

```python=
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(weather_3years.values, sunshine_tomorrow, test_size=0.3, random_state=0)
```

:bulb: **Tip**: you can have more than one list of data rows as a parameter to the [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) method from sklearn. See the documentation for more information.

We will have to normalize the input again as well. Now that we have prepared our data we can start training our model. Let's start with a simple linear regression model.

```python=
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
```

Construct the regression pipeline:

```python=
pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('regressionmodel', LinearRegression())
])

pipe
```

Train our model:

```python=
pipe.fit(X_train, y_train)
```

Measure the performance of the model:

**Note**: we cannot use a confusion matrix as before because we are not doing classification. The output of our prediction will be a continuous value.

```python=
pred_train= pipe.predict(X_train)
pred_test= pipe.predict(X_test)
```

In these cases we usually plot the predictions against the "ground truth" to get a feeling.

```python=
import matplotlib.pyplot as plt
plot = sns.scatterplot(x=y_train, y=pred_train)
```

Display and customise plot (training set):

```python=
plot.set_title('Performance on training set')
plot.set_xlabel('ground truth')
plot.set_ylabel('prediction')
plot.set_xbound((0, 16))
plot.set_ybound((-3, 16))

sns.lineplot(x=np.arange(16), y=np.arange(16), color='red')
```

Display and customise plot (test set):

```python=
plot = sns.scatterplot(x=y_test, y=pred_test)
plot.set_title('Performance on test set')
plot.set_xlabel('ground truth')
plot.set_ylabel('prediction')
plot.set_xbound((0, 16))
plot.set_ybound((-3, 16))

sns.lineplot(x=np.arange(16), y=np.arange(16), color='red')
```

We can also have a look at performance metrics like [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) and [mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error). The advantage of mean absolute error is that it is expressed in the same units as the prediction values.

Training set performance:

```python=
from sklearn.metrics import mean_squared_error, mean_absolute_error
display('training set')
display(f'MSE {mean_squared_error(y_train, pred_train)}  MAE: {mean_absolute_error(y_train, pred_train)}')
```

Test set performance:

```python=
display('test set')
display(f'MSE {mean_squared_error(y_test, pred_test)}  MAE: {mean_absolute_error(y_test, pred_test)}')
```

It is hard to say whether our results are good or bad. This depends on whether our prediction problem is hard or easy.

One way to evaluate can be to think of a very simple model and compare it against that.

One idea could be to use the number of sunshine hours of the current day to predict the sunshine hours the next day. Consecutive days can have similar weather, right?

Get the sunshine hours from the dataset and split them into training and test set:

```python=
baseline_idx = weather_3years.columns.get_loc('BASEL_sunshine')
baseline_train = X_train[:, baseline_idx]
baseline_test = X_test[:, baseline_idx]
```

Define function to plot performance of a given model against the baseline:

```python=
def plot_against_baseline(ground_truth, predicted, baseline, axis=None, label=''):
    plot = sns.scatterplot(x=ground_truth, y=predicted, label='prediction', alpha=0.5, ax=axis)
    plot.set_title(label)
    plot.set_xlabel('ground truth')
    plot.set_ylabel('prediction')
    plot.set_xbound((0, 16))
    plot.set_ybound((-3, 16))

    sns.scatterplot(x=ground_truth, y=baseline, label='baseline', alpha=0.5, ax=axis)
    sns.lineplot(x=[0, 16], y=[0,16], ax=axis)
    display(f'Baseline MSE: {mean_squared_error(ground_truth, baseline)}')
    display(f'Predicted MSE: {mean_squared_error(ground_truth, predicted)}')

```

Plot the performance of the model on the training (and test) set(s) against the baseline performance:

```python=
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
plot_against_baseline(y_train, pred_train, baseline_train, axis=axes[0], label='Performance on training set')
plot_against_baseline(y_test, pred_test, baseline_test, axis=axes[1], label='Performance on test set')
```

Saving your model:

```python=
import pickle

with open('lr_model.pkl', 'wb') as f:
    pickle.dump(pipe, f)
```

Re-opening / loading your model from file:

```python=
with open('lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)
```

Lets try other regression algorithms:
...See Exercises section

### Best practices in ML projects:

Before beginning your project, ask yourself:

+ What is your scientific problem? Is it clear what this is?
+ Can this problem be translated into a machine learning problem and, if so, how? Keep the problem simple or decompose it into smaller subproblems. What specific subtype of ML problem is it?
    + Supervised / Unsupervised?
    + Classification / Regression?
+ Do you even require machine learning to solve this problem?
+ What is the goal of your ML project?
+ Do you have enough **high-quality** data to solve this problem?
+ How will you measure the performance of your model?
    + First design and implement metrics to measure the performance
+ Do you have enough compute power (e.g. GPUs or CPUs) to train the model / storage space to store your data and models?
+ Are there ethical or privacy risks associated with the data or applying ML to it?

While conducting the ML project:

+ You **must** have a workflow. A bad one is better than no workflow!
    + Make one and optimize it (iteratively improve it)
+ Consider the different activities you will need in preprocessing, preparing and working with the data (excluding the core ML tasks such as training and evaluating your model). It is usual to spend the majority of your time, cleaning, preprocessing and exploring your data. So be very patient with data engineering.
+ Split data into training, validation and test sets
+ **NEVER** mix uses of data:
    + Training data should only be used for training
    + Validation data should only be used for validation (i.e. selecting a model to use)
    + Same for test data (only for estimating how well the performance of your model will generalise to unseen data)
+ Use common-sense features
+ Borrow features from other state-of-the-art models

Considerations about the model itself:

+ Set a baseline performance or model
    + Instead of training your own, you could also opt to use an existing state-of-the-art model as your baseline
    + human performance could also be used (how well does a human do at this task?)
    + guess it with your experience
+ Keep your first model simple. Getting it to work is half the success! Even if the performance is bad. Thereafter, you can try to iteratively improve it
+ Be patient with training, it is an iterative process to improve your model

Considerations after training:

+ Version your data and code (everything in your project folder!). This helps to quickly revert back to previous versions that at least work decently if you mess something up.
+ Re-train the model whenever possible by taking into account any new changes in the data or new features that are relevant

## 📚 Resources
- [Open ODISSEI eScience Call 2022](https://www.esciencecenter.nl/calls-for-proposals/open-odissei-escience-call-2022/)
    - [the slide showed during the lesson](https://nlesc.sharepoint.com/:p:/s/instructors/EXHnBm4JBOJKvkt4ByOuotYBIhrNEP1bgvoLAXAWDTvGhg?e=hjhxdx)
- [**Post-workshop survey**](https://www.surveymonkey.com/r/2CRDRBH)
- Tools to manage ML project (e.g. version data, code & model)
    - using [git and github](https://github.com/) 
    - [Weights & Biases](https://wandb.ai)
    - [MLFlow](https://mlflow.org/)
- [A famous paper about biased ML](https://proceedings.neurips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf)
- To learn more about ML
    - [A visual intro to ML](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
    - [Coursera Machine Learning Specialization by Andrew Ng](https://www.coursera.org/specializations/machine-learning-introduction)
    - [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, by Aurélien Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) - A logical, reasonably standardized, but flexible project structure for doing and sharing data science work.
- [The Turning Way](https://the-turing-way.netlify.app/welcome) - A handbook to reproducible, ethical and collaborative data science
- [deon](https://deon.drivendata.org/) - An ethics checklist for data scientists
