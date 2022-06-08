# odissei-machine-learning


## TOC 

Day 1
Morning
- Course intro (Dafne)
- Introduction slides (1h) (Dafne)
- NB1 (45min) (Sven)
- NB2 (30min) (Dafne)
- Theory on (binary) classification (45min) (Sven)
    - Decisiontree
    - Nearest neighbor
- NB3 (45min) (Sven)

Afternoon
- Theory on performance (20min) (Dafne
- NB4 (45min) (Dafne)
- Apply skills to LISS (Sven)
- Presentation Joris Mulder


Day 2
Morning
- Theory on regression (45min)
    - Linear regression
    - Neural net
- NB5 (30min)
- Feature selection
- Best practices

Afternoon
- Apply to LISS data (Sven)
- Presentation Wouter van Atteveldt


## Technical Requirements
A laptop with anaconda, python 3.9 and the latest versions of the following dependencies:
- scikit-learn
- pandas
- numpy
- matplotlib
- jupyter-notebook
- jupyterlab
- seaborn

## Potential table of content

- Introduction - **Slides to be created from [introduction content](https://github.com/esciencecenter-digital-skills/SICSS-odissei-machine-learning/blob/main/1-Intro.md)**
    -  What is ML
    -  AI, ML and DL
    -  ML and Statistics
    -  Types of ML
        -  Supervised learning
            -  Regression
            -  Classification
        -  Unsupervised learning
            -  Clustering
            -  Dimensionality Reduction
        -  Reinforcement learning
    - Limitations of machine learning
        - Data
        - Extrapolation
        - Interpretation of Results
    - Machine learning glossary
-  ML Workflow (with scikit-learn code) - **Adapt [notebook 1](https://github.com/esciencecenter-digital-skills/SICSS-odissei-machine-learning/blob/main/notebooks/1-Intro.ipynb)**
    -  Formulate / Outline the problem
    -  Identify inputs and outputs (data exploration)
        -  Intro Pandas, numpy, seaborn
        -  Data statistics and plots
        -  conversion (e.g. from Yes/No to 1/0)
    -  Prepare data (preprocessing)- **[notebook 2](https://github.com/esciencecenter-digital-skills/SICSS-odissei-machine-learning/blob/main/notebooks/2-Data-Preparation.ipynb)**
        -  check missing data
        -  clean data
        -  splitting data
    -  Choose an algorithm - **[notebook 3](https://github.com/esciencecenter-digital-skills/SICSS-odissei-machine-learning/blob/main/notebooks/3-Model-pipeline.ipynb)**
        -  Use sklearn.dummy.DummyRegressor
    -  Train the model - **[notebook 3](https://github.com/esciencecenter-digital-skills/SICSS-odissei-machine-learning/blob/main/notebooks/3-Model-pipeline.ipynb)**
    -  Perform a Prediction/Classification (applying the model) - **[notebook 3](https://github.com/esciencecenter-digital-skills/SICSS-odissei-machine-learning/blob/main/notebooks/3-Model-pipeline.ipynb)**
    -  Measure performance (validate the model) - **[notebook 4](https://github.com/esciencecenter-digital-skills/SICSS-odissei-machine-learning/blob/main/notebooks/4-CrossValidation.ipynb)**
        -  Cross validation
    -  Save model
-  Regression example - **Create slides on models**
    -  Ordinary Least squares
    -  SVM
-  Classification example - **Create slides on models**
    -  Nearest neighbors
    -  Decision trees
        -  Random forest
-  Metrics- **Create slides**
    -  Classification
        -  F1 score
            -  Accuracy
        -  Confusion matrix
        -  ROC
    -  Regression
-  Feature selection / dimensionality reduction - **Create notebook**
    -  Cross correlation
    -  PCA
    -  tSNE
-  Hyper-parameter optimizers  - **[notebook 4](https://github.com/esciencecenter-digital-skills/SICSS-odissei-machine-learning/blob/main/notebooks/4-CrossValidation.ipynb)**
    -  sk-learn.model_selection.GridSearchCV
-  ML algorithms
    -  Nearest neighbors
    -  Ordinary Least squares
    -  Logistic regression
    -  Naïve Bayes
    -  Decision trees
    -  Random forest
    -  SVM
    -  Neural net
        -  Single-layer perceptron
        -  Multi-layer perceptron
-  Best practices
-  Exercise (+Q&A, whole afternoon)
    -  Setup own experiment (with their own dataset and questions)
-  Useful resources
    - [ML Map](https://scikit-learn.org/stable/_static/ml_map.png)
    - [Data Science cookiecutter](https://drivendata.github.io/cookiecutter-data-science/)
    - [The Turning Way](https://the-turing-way.netlify.app/welcome)
