from packages.linear_model import MulticlassLogisticRegression as mlr
import numpy as np
from scipy.sparse import csr_matrix


def test__calculate_number_classes__one_class():
    y = np.array([0, 0, 0])
    assert mlr._calculate_number_classes(y) == 1


def test__calculate_number_classes__multiple_class():
    y = np.array([0, 1, 2, 1, 0])
    assert mlr._calculate_number_classes(y) == 3


def test_convert_to_binary_classes__index0():
    y = np.array([0, 1, 2, 1, 0])

    expected = np.array([1, 0, 0, 0, 1])
    actual = mlr._convert_to_binary_classes(y, 0)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(y, y)  # ensure original array is unchanged


def test_convert_to_binary_classes__index1():
    y = np.array([0, 1, 2, 1, 0])

    expected = np.array([0, 1, 0, 1, 0])
    actual = mlr._convert_to_binary_classes(y, 1)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(y, y)  # ensure original array is unchanged


def test_convert_to_binary_classes__index2():
    y = np.array([0, 1, 2, 1, 0])

    expected = np.array([0, 0, 1, 0, 0])
    actual = mlr._convert_to_binary_classes(y, 2)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(y, y)  # ensure original array is unchanged


def test__standardize_probabilities():
    y_pred = np.array([[0.59652226, 0.1826289, 0.19549276],
                       [0.00964624, 0.95665144, 0.00964272],
                       [0.22047453, 0.22440378, 0.84527293],
                       [0.2111439, 0.14770427, 0.32313616]])

    expected = np.array([[0.61204123, 0.18738013, 0.20057865],
                         [0.00988405, 0.98023551, 0.00988044],
                         [0.17089045, 0.17393603, 0.65517352],
                         [0.30960227, 0.21658015, 0.47381757]])

    actual = mlr._standardize_probabilities(y_pred)
    np.testing.assert_allclose(actual, expected, atol=1e-7)

    # ensure all rows add up to one
    np.testing.assert_allclose(np.sum(actual, axis=1), np.array([1., 1., 1., 1.]), atol=1e-16)


def test__update_predictions():
    y_pred = np.array([1, 1, 1])
    y_pred_proba_k = np.array([0.6, 0.4, 0.8])
    y_pred_proba_highest = np.array([0.5, 0.6, 0.7])
    k = 3

    mlr._update_predictions(y_pred, y_pred_proba_highest, y_pred_proba_k, k)
    y_pred_expected = np.array([3, 1, 3])
    y_pred_proba_highest_expected = np.array([0.6, 0.6, 0.8])

    np.testing.assert_array_equal(y_pred, y_pred_expected)
    np.testing.assert_array_equal(y_pred_proba_highest, y_pred_proba_highest_expected)


def test__init__v2():
    eta = 1
    epsilon = 2
    model = mlr.MulticlassLogisticRegression2(eta, epsilon)
    assert model.eta == eta
    assert model.epsilon == epsilon
    assert model.classifiers is None


def test_fit__v2():
    X = np.array([[1, 1],
                  [0, 0],
                  [1, 0],
                  [0, 4],
                  [5, 1],
                  [5, 2],
                  [5, -1],
                  [5, 10],
                  [3, 10],
                  [3, 10.5],
                  [3, 11]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    model = mlr.MulticlassLogisticRegression2(eta=0.01, epsilon=0.01)
    model.fit(X, y)
    assert len(model.classifiers) == 3


def test_fit__v2__sparse():
    X = csr_matrix(np.array([[1, 1],
                             [0, 0],
                             [1, 0],
                             [0, 4],
                             [5, 1],
                             [5, 2],
                             [5, -1],
                             [5, 10],
                             [3, 10],
                             [3, 10.5],
                             [3, 11]]))
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    model = mlr.MulticlassLogisticRegression2(eta=0.01, epsilon=0.01)
    model.fit(X, y)
    assert len(model.classifiers) == 3


def test_predict_proba_v2():
    X = np.array([[1, 1],
                  [0, 0],
                  [1, 0],
                  [0, 4],
                  [5, 1],
                  [5, 2],
                  [5, -1],
                  [5, 10],
                  [3, 10],
                  [3, 10.5],
                  [3, 11]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    model = mlr.MulticlassLogisticRegression2(eta=0.01, epsilon=0.01)
    model.fit(X, y)

    X_test = np.array([[0, 1],
                       [5, 0],
                       [3, 10.25],
                       [0, 5]])

    expected = np.zeros((3, 4))
    for k, lr in enumerate(model.classifiers):
        predict_proba_k = lr.predict_proba(X_test)[:, 1]
        expected[k] = predict_proba_k
    sums = np.sum(expected.T, axis=1).reshape(-1, 1)
    standardized = expected.T / sums

    actual = model.predict_proba(X_test)
    np.testing.assert_allclose(actual, standardized, atol=1e-16)


def test_predict_proba_v2__sparse():
    X = csr_matrix(np.array([[1, 1],
                             [0, 0],
                             [1, 0],
                             [0, 4],
                             [5, 1],
                             [5, 2],
                             [5, -1],
                             [5, 10],
                             [3, 10],
                             [3, 10.5],
                             [3, 11]]))
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    model = mlr.MulticlassLogisticRegression2(eta=0.01, epsilon=0.01)
    model.fit(X, y)

    X_test = csr_matrix(np.array([[0, 1],
                                  [5, 0],
                                  [3, 10.25],
                                  [0, 5]]))

    expected = np.zeros((3, 4))
    for k, lr in enumerate(model.classifiers):
        predict_proba_k = lr.predict_proba(X_test)[:, 1]
        expected[k] = predict_proba_k
    sums = np.sum(expected.T, axis=1).reshape(-1, 1)
    standardized = expected.T / sums

    actual = model.predict_proba(X_test)
    np.testing.assert_allclose(actual, standardized, atol=1e-16)


def test_predict_v2():
    X = np.array([[1, 1],
                  [0, 0],
                  [1, 0],
                  [0, 4],
                  [5, 1],
                  [5, 2],
                  [5, -1],
                  [5, 10],
                  [3, 10],
                  [3, 10.5],
                  [3, 11]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    model = mlr.MulticlassLogisticRegression2(eta=0.01, epsilon=0.01)
    model.fit(X, y)

    X_test = np.array([[0, 1],
                       [5, 0],
                       [3, 10.25],
                       [0, 5]])

    expected = np.array([0, 1, 2, 2])
    actual = model.predict(X_test)
    np.testing.assert_array_equal(actual, expected)


def test_predict_v2__sparse():
    X = csr_matrix(np.array([[1, 1],
                             [0, 0],
                             [1, 0],
                             [0, 4],
                             [5, 1],
                             [5, 2],
                             [5, -1],
                             [5, 10],
                             [3, 10],
                             [3, 10.5],
                             [3, 11]]))
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    model = mlr.MulticlassLogisticRegression2(eta=0.01, epsilon=0.01)
    model.fit(X, y)

    X_test = csr_matrix(np.array([[0, 1],
                                  [5, 0],
                                  [3, 10.25],
                                  [0, 5]]))

    expected = np.array([0, 1, 2, 2])
    actual = model.predict(X_test)
    np.testing.assert_array_equal(actual, expected)


def test_predict_v2__l2_penalty():
    X = np.array([[1, 1],
                  [0, 0],
                  [1, 0],
                  [0, 4],
                  [5, 1],
                  [5, 2],
                  [5, -1],
                  [5, 10],
                  [3, 10],
                  [3, 10.5],
                  [3, 11]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    penalty = 'l2'
    penalty_lambda = 0.1
    model = mlr.MulticlassLogisticRegression2(eta=0.01,
                                              epsilon=0.01,
                                              penalty=penalty,
                                              l2_lambda=penalty_lambda)
    model.fit(X, y)

    X_test = np.array([[0, 1],
                       [5, 0],
                       [3, 10.25],
                       [0, 5]])

    expected = np.array([0, 1, 2, 2])
    actual = model.predict(X_test)
    np.testing.assert_array_equal(actual, expected)


def test_predict_v2__l2_penalty__sparse():
    X = csr_matrix(np.array([[1, 1],
                             [0, 0],
                             [1, 0],
                             [0, 4],
                             [5, 1],
                             [5, 2],
                             [5, -1],
                             [5, 10],
                             [3, 10],
                             [3, 10.5],
                             [3, 11]]))
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    penalty = 'l2'
    penalty_lambda = 0.1
    model = mlr.MulticlassLogisticRegression2(eta=0.01,
                                              epsilon=0.01,
                                              penalty=penalty,
                                              l2_lambda=penalty_lambda)
    model.fit(X, y)

    X_test = csr_matrix(np.array([[0, 1],
                                  [5, 0],
                                  [3, 10.25],
                                  [0, 5]]))

    expected = np.array([0, 1, 2, 2])
    actual = model.predict(X_test)
    np.testing.assert_array_equal(actual, expected)
