import packages.LogisticRegression.LogisticRegression as lr
import numpy as np


def test__set_weights():
    X = np.array([[1, 2, 3],
                  [1, 2, 3]])
    expected = np.array([0, 0, 0])
    actual = lr._set_weights(X)
    np.testing.assert_array_equal(actual, expected)


def test__add_x0_one_col_multiple_samples():
    X_test = np.array([[1],
                       [2],
                       [3]])
    actual = lr._add_x0(X_test)
    expected = np.array([[1, 1],
                         [1, 2],
                         [1, 3]])
    np.testing.assert_array_equal(actual, expected)


def test__add_x0_two_cols_multiple_samples():
    X_test = np.array([[1, 9],
                       [2, 7],
                       [3, 8]])
    actual = lr._add_x0(X_test)
    expected = np.array([[1, 1, 9],
                         [1, 2, 7],
                         [1, 3, 8]])
    np.testing.assert_array_equal(actual, expected)


def test__init__():
    a = lr.LogisticRegression(eta=0.01, epsilon=0.5)
    assert a.eta == 0.01
    assert a.epsilon == 0.5
    assert a.weights is None


def test_predict_two_samples():
    a = lr.LogisticRegression(eta=0.01, epsilon=0.5)
    a.weights = np.array([.1,.2,.3])
    X = np.array([[4, 5],[3, 5]])

    expected = np.array([1,1])
    actual = a.predict(X)
    np.testing.assert_array_equal(actual, expected)
