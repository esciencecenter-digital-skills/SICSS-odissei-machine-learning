from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def test_numpy():
    a = np.arange(4)

    assert a.sum() == 6


def test_pandas():
    df = pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])

    assert df['a'].sum() == 4


def test_matplotlib():
    # Just running this one without errors should be enough
    x = [1, 2, 3, 4]
    plt.scatter(x=x, y=[el * el for el in x])


def test_seaborn():
    x = [1, 2, 3, 4]

    sns.barplot(x=x, y=x)


def test_sklearn():
    def fun(x):
        return 2 * x + 3

    X = np.arange(10).reshape((10, 1))
    y = fun(X)

    model = LinearRegression()

    model.fit(X, y)

    np.testing.assert_array_almost_equal(model.coef_, [[2.0]], )


if __name__ == '__main__':
    test_numpy()
    test_pandas()
    test_matplotlib()
    test_seaborn()
    test_sklearn()
    print('Your environment is has been correctly set up!')

