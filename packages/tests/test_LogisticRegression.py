import packages.linear_model.LogisticRegression as lr
import numpy as np
from scipy.sparse import csr_matrix


def test__set_weights():
    X = np.array([[1, 2, 3],
                  [1, 2, 3]])
    expected = np.array([0, 0, 0])
    actual = lr._set_weights(X)
    np.testing.assert_array_equal(actual, expected)


def test__set_weights__sparse():
    X = csr_matrix(np.array([[1, 2, 3],
                             [1, 2, 3]]))
    expected = np.array([0, 0, 0])
    actual = lr._set_weights(X)
    np.testing.assert_array_equal(actual, expected)


def test__add_x0_one_col_multiple_samples():
    X_test = np.array([[1],
                       [2],
                       [3]])
    actual = lr._add_x0(X_test).toarray()
    expected = np.array([[1, 1],
                         [1, 2],
                         [1, 3]])
    np.testing.assert_array_equal(actual, expected)


def test__add_x0_one_col_multiple_samples__sparse():
    X_test = csr_matrix(np.array([[1],
                                  [2],
                                  [3]]))
    actual = lr._add_x0(X_test).toarray()
    expected = np.array([[1, 1],
                         [1, 2],
                         [1, 3]])
    np.testing.assert_array_equal(actual, expected)


def test__add_x0_two_cols_multiple_samples():
    X_test = np.array([[1, 9],
                       [2, 7],
                       [3, 8]])

    expected = np.array([[1, 1, 9],
                         [1, 2, 7],
                         [1, 3, 8]])
    actual = lr._add_x0(X_test).toarray()
    np.testing.assert_array_equal(actual, expected)


def test__add_x0_two_cols_multiple_samples__sparse():
    X_test = csr_matrix(np.array([[1, 9],
                                  [2, 7],
                                  [3, 8]]))

    expected = np.array([[1, 1, 9],
                         [1, 2, 7],
                         [1, 3, 8]])
    actual = lr._add_x0(X_test).toarray()
    np.testing.assert_array_equal(actual, expected)


def test__init__():
    a = lr.LogisticRegression(eta=0.01,
                              epsilon=0.5,
                              penalty='l2',
                              l2_lambda=5)
    assert a.eta == 0.01
    assert a.epsilon == 0.5
    assert a.weights is None
    assert a.penalty == 'l2'
    assert a.l2_lambda == 5


def test_fit():
    a = lr.LogisticRegression(eta=0.01, epsilon=0.5)
    X = np.array([[4, 5], [3, 5]])
    y = np.array([1, 0])
    a.fit(X, y)
    assert a.weights is not None


def test_fit__sparse():
    a = lr.LogisticRegression(eta=0.01, epsilon=0.5)
    X = csr_matrix(np.array([[4, 5], [3, 5]]))
    y = np.array([1, 0])
    a.fit(X, y)
    assert a.weights is not None


def test_predict_two_samples():
    a = lr.LogisticRegression(eta=0.01, epsilon=0.5)
    a.weights = np.array([.1, .2, .3])
    X = np.array([[4, 5], [3, 5]])

    expected = np.array([1, 1])
    actual = a.predict(X)
    np.testing.assert_array_equal(actual, expected)


def test_predict_two_samples__sparse():
    a = lr.LogisticRegression(eta=0.01, epsilon=0.5)
    a.weights = np.array([.1, .2, .3])
    X = csr_matrix(np.array([[4, 5], [3, 5]]))

    expected = np.array([1, 1])
    actual = a.predict(X)
    np.testing.assert_array_equal(actual, expected)


def test_predict_proba():
    a = lr.LogisticRegression(eta=0.01, epsilon=0.5)
    a.weights = np.array([.1, .2, .3])
    X = np.array([[4, 5], [3, 5]])
    a.predict_proba(X)


def test_predict_proba__sparse():
    a = lr.LogisticRegression(eta=0.01, epsilon=0.5)
    a.weights = np.array([.1, .2, .3])
    X = csr_matrix(np.array([[4, 5], [3, 5]]))
    a.predict_proba(X)
