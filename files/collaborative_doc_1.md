![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 1

2022-06-22 SICSS-ODISSEI Machine learning.

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [link](https://tinyurl.com/274bduec)

Collaborative Document day 1: [link]([<url>](https://tinyurl.com/274bduec))

Collaborative Document day 2: [link](https://tinyurl.com/mspb2kfk)


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

To ask a question during the lecture, please raise your hand.

For getting help during the exercises we use a post-it system. If you need help, please put the blue post-it on top of your laptop, and one of the helpers will come to help you. When you are doing fine with the exercises, please have the yellow post-it on your laptop.

General tips and tricks:
In Jupyter lab, you can use Shift+Tab in a cell to get help on a function.

## 🖥 Links

Download files 
[Weather dataset](https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1)
[BBQ weather labels](https://zenodo.org/record/5071376/files/weather_prediction_bbq_labels.csv?download=1)

## 👩‍🏫👩‍💻🎓 Instructors

Sven, Dafne, Cunliang, Djura

## 🧑‍🙋 Helpers

Christiaan, Kody, Laura

## 🗓️ Agenda
| Time | Topic |
|--:|:---|
| 9:00 | 	Welcome and Introduction |
| 10:00 | Live coding: introduction to weather dataset  |
| 10:45| Coffee break |
| 11:00 | Live coding: data preparation |
| 11:30 | Theory on binary classification |
|12:15|Lunch|
|12:45|Live coding: building a classifier|
|13:30| Theory on performance|
|13:50| Live coding: cross validation and hyperparameter optimization|
|14:35| Coffee break|
|14:50| Apply skills to LISS|


## 🔧 Exercises
### Supervised or unsupervised?
For the following problems, do you think you need a supervised or unsupervised approach? 
Put up your Post-it to answer
Yellow = supervised
Blue = unsupervised

- Find numerical representations for words in a language (word vectors) that contain semantic information on the word
- Determine whether a tumor is benign or malign, based on an MRI-scan
- Predict the age of a patient, based on an EEG-scan
- Cluster observations of plants into groups of individuals that have similar properties, possibly belonging to the same species

### Excercise 1: explore the weather data set
Explore the dataset with pandas:
 1. How many features do we have to predict the BBQ weather? 
 2. What data type is the target label stored in?
 3. How many samples does this dataset have?
 4. (Optional): How many features do we have in each city?

```python=
sum(1 for col in df_feat.columns if col.startswith('BASEL')) #crappy solution for 1 of them.

#For each one:
collist2 = []
for col in df_feat.columns:
    col = col.split("_")[0]
    collist2.append(col)
collist2 = collist2[2:]
[[x,collist2.count(x)] for x in set(collist2)]
```


### Exercise: Create the pairplot using Seaborn

Discuss what you see in the scatter plots and write down any observations. Think of:

1. Are the classes easily separable based on these features?
2. What are potential difficulties for a classification algorithm?
3. What do you note about the units of the different attributes?


### Data cleaning and preparation
### Exercise 1
Compare the distributions of the numerical features before and after scaling. What do you notice?



### Model pipeline
#### Exercise
Discuss in groups: What do you observe in the confusion matrix? Does the confusion matrix help you decide whether the model will perform well enough on new data?

### Assignment
Predict income and contract type in 2021 based on the data from 2011.

In groups of 2 people:
1. Pick a problem:
* Predict gross salary in 2021 (regression problem)
* Predict contract type in 2021 (classification problem)
2. Solve the problem as best as you can

Example steps to take (for today, tomorrow and Friday):
- Define the problem
- Exploratory data analysis -> prepare the data
- Train and evaluate your first model
- Iterate over the machine learning cycle! Try out different approaches to find the best one.
- Train and evaluate both models
- Compare predictions

Protip for categorical variables: [pd.get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html). This is only needed for dependent variables (features). Most sklearn classifiers handle multiclass categorical targets. You can check out some documentation on sklearn multiclass algoritms [here](https://scikit-learn.org/stable/modules/multiclass.html). 

## 🧠 Collaborative Notes
### Getting to know the data
#### Open a Jupyter notebook

Please open a Jupyter notebook so you can follow along :)

On the command line (e.g. a terminal) you can usually do this by typing:
```
jupyter-lab
```
This should open in your web browser. Select a new python notebook.


```bash=
# Getting to know the data
```
#### Learning objectives:

* Get to know the weather prediciton dataset
* Know the steps in the machine learning workflow
* Know how to do exploratory data analysis
* Know how to split the data in train and test set

#### Machine learning workflow

For most machine learning approaches, we have to take the following steps:

1. Data cleaning and preperation
2. Split data into train and test set
3. Optional: Feature selection
4. Use cross validation to:
    * Train one or more ML models on the train set
    * Choose optimal model / parameter settings based on some metric
5. Calculate final model performance on the test set

Import the following Python packages in your notebook:
```python =
import seaborn as sns
import pandas as pd
```

Read in the data:
``` python=
url_features = 'https://zenodo.org/record/5071376/files/weather_prediction_dataset.csv?download=1'
url_labels = 'https://zenodo.org/record/5071376/files/weather_prediction_bbq_labels.csv?download=1'
```
``` python=
weather_features = pd.read_csv(url_features)
weather_labels = pd.read_csv(url_labels)
```

To show the first lines of the data and see which columns there are:
```python= 
weather_features.head()
```
To check out the dimensions of the data:
```python=
wheater_label.shape()
```

Our task: predict whether we can BBQ tomorrow. This is a binary classification problem, because we have two possible outcomes: yes or no.

```python =
# 1 How many features do we have to predict the BBQ weather?
weather_features.shape()
```

```python =
# 2 What data type is the target label stored in?
type(weather_labels['BASEL_BBQ_weather'][0])
```

```python =
# 3 How many samples does this dataset have?
# Again use:
weather_features.shape()
```

### Data Selection
``` python
# Define how many rows make up 3 years of data
nr_rows = 365*3
```
Select the data we want to use:
``` python
# Drop columns we don't need
weather_dropped = weather_features.drop(columns=['DATE', 'MONTH'])
# Select first 3 years (assuming we have one row per day (ordered))
weather_3years = weather_dropped[:nr_rows]
```
Let's calculate an extra column: for each day, get the data on if it is BBQ weather in Basel on the next day.
Reminder: Python starts counting at 0 by default (i.e. the first item of a list is located in position 0)
``` python
# Create new column
weather_3years['BASEL_BBQ_weather'] = list(weather_labels[1:nr_rows+1]['BASEL_BBQ_weather'])
# Check out your updated data set:
weather_3years.head()
```

### Split the data into training and test set

``` python
# import specific function
from sklearn.model_selection import train_test_split
```

``` python
# split the data: 30% goes in the test set, 70% in the train set
data_train, data_test = train_test_split(weather_3years, test_size= 0.3, random_state=0)
# Important: we use random_state=0 to ensure we always all have the same train and test data
```

``` python
# check out how many data points are in the train and test set
len(data_train), len(data_test)
```
Write the data to csv files for future reference.
``` python
# Create a folder called data (if it does not exist already)
import os

if not os.path.exists('data'):
    os.mkdir('data')
```

``` python
data_train.to_csv('data/weather_features_train.csv', index=False)
data_test.to_csv('data/weather_features_test.csv', index=False)
```

### Visualisation
``` python
# List comprehension
columns = [c for c in data_train.columns if c.startswith('BASEL')]
columns
```

``` python
data_to_plot = data_train[columns]
```

``` python
data_to_plot.head(2)
```

``` python
# Plot the data using pairplot (plots each feature as a function of each of the other features. 
# Different colours indicate the value of BASEL_BBQ_weather)
sns.pairplot(data_to_plot, hue='BASEL_BBQ_weather')
```

### Key takeaways:

* Use the machine learning workflow to tackle a machine learning problem in a structured way
* Get to know the dataset before diving into a machine learning task
* Use an independent testset for evaluating your results

### Data cleaning and preparation
Check for missing data
```python
# check for missing data per column
data_train.isna().sum()
```

```python
# check for missing data in all columns combined
data_train.isna().sum().sum()
```

Note: there are various methods to normalise your data, here we will use the min-max method to get all data scale to be in the range 0-1.
```python
import sklearn.preprocessing
```

```python
# Feature normalisation using min-max scaling
min_max_scaler = sklearn.preprocessing.MinMaxScaler()
```

```python
# select all but the last column names, such that we can 
# (in the next step) apply the normalisation to the numerical values only
feature_names = data_train.columns[:-1]
```

```python
# create new data frame
data_train_scaled = data_train.copy()
data_train_scaled[feature_names] = min_max_scaler.fit_transform(data_train[feature_names])
```

```python
data_train_scaled.head()
```

``` python
data_train_scaled.describe()
```

```python
# Save the data
data_train_scaled.to_csv('data/weather_train_scaled.csv', index=False)
```
### Classification
#### Learning objectives:

* Be able to train a classifier using scikitlearn
* Know how to make predictions using a trained scikitlearn classifier
* Explain how a few examplary classificaion algorithms work: k-nearest neighbours, decision trees and random forests
* Know how to use scikitlearn's Pipeline to easily create machine learning pipelines

We start with a new notebook. This means that we have to import the libraries we want to use and load the data again.

```python
# read in the data we saved earlier
import pandas as pd
weather_train_scaled = pd.read_csv('data/weather_train_scaled.csv')
```

```python
weather_train_scaled.head(2)
```

```python
# obtain the scaled column names
features = weather_train_scaled.columns[:-1]
```

```python

---

X = weather_train_scaled[features]
y = weather_train_scaled['BASEL_BBQ_weather']
```

### Understanding the KNearestNeighboursClassifier
```python
from sklearn.neighbors import KNeighborsClassifier
```

```python
# First we define the model and its parameters
classifier = KNeighborsClassifier(n_neighbors=3)
# Now we can train it by passing it our data
classifier.fit(X, y)
```

```python
# We want to create a plot to get an intuition for how KNearestNeighboursClassifier works
# Let's plot two features against each other (I picked these based on the pairplot from yesterday).
# Let's add colourcoding for the different labels
import seaborn as sns
_ = sns.scatterplot(data=weather_train_scaled,
                    x='BASEL_temp_mean',
                    y='BASEL_global_radiation',
                    hue='BASEL_BBQ_weather'
                    )
```

### Understanding the Decision Tree Classifier
```python
from sklearn.tree import DecisionTreeClassifier
```

```python
# we define the model
classifier = DecisionTreeClassifier(max_leaf_nodes=6)
classifier.fit(X, y)
```
max_leaf_nodes is a hyperparameter that can be fine-tuned. Setting it to a value that is too large will lead to overfitting. With a value that is too small, you may not be able to catch to full complexity of the data

```python
# Plotting the decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
# To get an intution for decision trees, we are going to visualize the decision tree
# We will create a function so we can reuse it later
def plot_decision_tree(decision_tree_classifier):
    _ = plot_tree(decision_tree_classifier, 
                  filled=True,
                  feature_names=features,
                  class_names=['NO BBQ tomorrow', "Let's BBQ tomorrow"]
    )
_, ax = plt.subplots(figsize=(20, 20)) # This is to make it readable
plot_decision_tree(classifier)
```
### Let's train another tree and see how it performs
```python
classifier = DecisionTreeClassifier(min_samples_leaf=2)
classifier.fit(X, y)
```
```python
predictions = classifier.predict(X)
```

```python
predictions[:10]
```

```python
from sklearn.metrics import ConfusionMatrixDisplay
ConFusionMatrixDisplay.from_estimator(classifier, X, y)
```
### Pipelines
``` python
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
```

```python
# load the saved training data
weather_train = pd.read_csv('data/weather_train.csv')
```

```python
X = weather_train[features]
y = weather_train['BASEL_BBQ_weather']
```

```python
# load the saved test data
weather_test = pd.read_csv('data/weather_test.csv')
```

```python
X_test = weather_test[features]
y_test = weather_test['BASEL_BBQ_weather']
```

```python
# Create a new pipeline object
pipe = Pipeline([
    ('scale', MinMaxScaler()),
    ('model', DecisionTreeClassifier(min_samples_leaf=2, random_state=0))
])
```
```python
pipe.fit(X, y)
```
```python
ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test)
```

```python
from sklearn.metrics import accuracy_score
```

```python
pred_test = pipe.predict(X_test)
```

```python
pred_train = pipe.predict(X)
```

```python
print('Train accuracy:', accuracy_score(y, pred_train))
print('Test accuracy:', accuracy_score(y_test, pred_test))
```

```python
# To check if our test accuracy is any good, 
# compare it to the accuracy if we always predict False
accuracy_score(y_test, [False]*len(y_test))
```

## Questions
List here questions you still have:

## 📚 Resources
- https://www.esciencecenter.nl/
- [Choosing the right estimator](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
