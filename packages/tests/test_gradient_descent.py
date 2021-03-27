import numpy as np
import packages.LogisticRegression.gradient_descent as gd


def test__calc_inner():
    X = np.array([[1, 5, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 2, 3]])
    w = np.array([2, 4, 5, 6])

    expected = np.array([33, 17, 34])
    actual = gd._calc_inner(X, w)
    np.testing.assert_array_equal(actual, expected)


def test__get_y_predictions():
    X = np.array([[.1, .5, .2, .1],
                  [.1, .1, .2, .1],
                  [.1, .1, .2, .5]])
    w = np.array([2, 4, 5, 6])

    expected = np.exp(np.matmul(X, w)) / (np.ones(3) + np.exp(np.matmul(X, w)))
    actual = gd.get_y_predictions(X, w)
    np.testing.assert_array_equal(actual, expected)


def test__calc_gradient():
    X = np.array([[1, 1, 1, 1],
                  [1, 5, 1, 1],
                  [1, 1, 2, 3]])
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 1, 0])
    expected = np.array([0, -4, 1, 2])
    actual = gd._calc_gradient(X, y_true, y_pred)
    np.testing.assert_array_equal(actual, expected)


def test__update_weights():
    eta = 0.01
    gradient = np.array([1, -2, 3])
    w = np.array([4, 5, 6])

    expected = np.array([4.01, 4.98, 6.03])
    actual = gd._update_weights(w, eta, gradient)
    np.testing.assert_array_equal(actual, expected)


def test__calc_left_half_log_likelihood():
    X = np.array([[1, 1, 1, 1],
                  [1, 5, 1, 1],
                  [1, 1, 2, 3]])
    w = np.array([2, 4, 5, 6])
    y_true = np.array([1, 0, 0])
    expected = 17
    actual = gd._calc_left_half_log_likelihood(X, y_true, w)
    assert actual == expected


def test__calc_right_half_log_likelihood():
    X = np.array([[1, 1, 1, 1],
                  [1, 5, 1, 1],
                  [1, 1, 2, 3]])
    w = np.array([2, 4, 5, 6])
    expected = 84.00000004139937
    actual = gd._calc_right_half_log_likelihood(X, w)
    assert actual == expected


def test__calc_log_likelihood():
    X = np.array([[1, 1, 1, 1],
                  [1, 5, 1, 1],
                  [1, 1, 2, 3]])
    w = np.array([2, 4, 5, 6])
    y_true = np.array([1, 0, 0])
    expected = 17 - 84.00000004139937
    actual = gd._calc_log_likelihood(X, y_true, w)
    assert actual == expected
