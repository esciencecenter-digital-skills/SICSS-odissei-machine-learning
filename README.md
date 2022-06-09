# odissei-machine-learning

## TOC

Day 1
Morning

- Introduction slides (1h)
- NB1 (45min)
- NB2 (30min)
- Theory on (binary) classification (45min)
    - Decisiontree
    - Nearest neighbor
- NB3 (45min)

Afternoon

- Theory on performance (20min)
- NB4 (45min)
- Apply skills to LISS
- Presentation Joris Mulder

Day 2
Morning

- Theory on regression (45min)
    - Linear regression
    - Neural net
- NB5 (30min)
- NB6 - Feature selection
- Best practices

Afternoon

- Apply to LISS data
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

### Setup instructions

To be honest, any recent version of python and the aforementioned list of dependencies will probably
work fine. However, if you are running into problems, the instructions below should give you a
working setup.

You will need to have  [anaconda](https://www.anaconda.com/) installed. The website will provide instructions for your
operating system.

Open a terminal, (command prompy or powershell on windows), and clone our setup git repo:

```bash
git clone https://github.com/esciencecenter-digital-skills/SICSS-setup.git
```

Then install the conda environment as follows:
```bash
cd SICSS-setup
conda env create -f environment.yml
```

Now activate this conda environment:
```bash
conda activate 
```

To check if your environment is running correclty, you can run our test script:
```bash
python check_setup.py
```
It should output `Your environment is has been correctly set up!` if it ran succesfully.

## Potential table of content

- Introduction - **Slides to be created
  from [introduction content](https://github.com/esciencecenter-digital-skills/SICSS-odissei-machine-learning/blob/main/1-Intro.md)**
    - What is ML
    - AI, ML and DL
    - ML and Statistics
    - Types of ML
        - Supervised learning
            - Regression
            - Classification
        - Unsupervised learning
            - Clustering
            - Dimensionality Reduction
        - Reinforcement learning
    - Limitations of machine learning
        - Data
        - Extrapolation
        - Interpretation of Results
    - Machine learning glossary
- ML Workflow (with scikit-learn code) - **
  Adapt [notebook 1](https://github.com/esciencecenter-digital-skills/SICSS-odissei-machine-learning/blob/main/notebooks/1-Intro.ipynb)**
    - Formulate / Outline the problem
    - Identify inputs and outputs (data exploration)
        - Intro Pandas, numpy, seaborn
        - Data statistics and plots
        - conversion (e.g. from Yes/No to 1/0)
    - Prepare data (preprocessing)
      - **[notebook 2](https://github.com/esciencecenter-digital-skills/SICSS-odissei-machine-learning/blob/main/notebooks/2-Data-Preparation.ipynb)**
        - check missing data
        - clean data
        - splitting data
    - Choose an algorithm
      - **[notebook 3](https://github.com/esciencecenter-digital-skills/SICSS-odissei-machine-learning/blob/main/notebooks/3-Model-pipeline.ipynb)**
        - Use sklearn.dummy.DummyRegressor
    - Train the model
      - **[notebook 3](https://github.com/esciencecenter-digital-skills/SICSS-odissei-machine-learning/blob/main/notebooks/3-Model-pipeline.ipynb)**
    - Perform a Prediction/Classification (applying the model)
      - **[notebook 3](https://github.com/esciencecenter-digital-skills/SICSS-odissei-machine-learning/blob/main/notebooks/3-Model-pipeline.ipynb)**
    - Measure performance (validate the model)
      - **[notebook 4](https://github.com/esciencecenter-digital-skills/SICSS-odissei-machine-learning/blob/main/notebooks/4-CrossValidation.ipynb)**
        - Cross validation
    - Save model
- Regression example - **Create slides on models**
    - Ordinary Least squares
    - SVM
- Classification example - **Create slides on models**
    - Nearest neighbors
    - Decision trees
        - Random forest
- Metrics- **Create slides**
    - Classification
        - F1 score
            - Accuracy
        - Confusion matrix
        - ROC
    - Regression
- Feature selection / dimensionality reduction - **Create notebook**
    - Cross correlation
    - PCA
    - tSNE
- Hyper-parameter optimizers
  - **[notebook 4](https://github.com/esciencecenter-digital-skills/SICSS-odissei-machine-learning/blob/main/notebooks/4-CrossValidation.ipynb)**
    - sk-learn.model_selection.GridSearchCV
- ML algorithms
    - Nearest neighbors
    - Ordinary Least squares
    - Logistic regression
    - Naïve Bayes
    - Decision trees
    - Random forest
    - SVM
    - Neural net
        - Single-layer perceptron
        - Multi-layer perceptron
- Best practices
- Exercise (+Q&A, whole afternoon)
    - Setup own experiment (with their own dataset and questions)
- Useful resources
    - [ML Map](https://scikit-learn.org/stable/_static/ml_map.png)
    - [Data Science cookiecutter](https://drivendata.github.io/cookiecutter-data-science/)
    - [The Turning Way](https://the-turing-way.netlify.app/welcome)
