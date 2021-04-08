from packages.LogisticRegression import MulticlassLogisticRegression as mlr
import numpy as np


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


def test__add_x0_one_col_multiple_samples():
    X_test = np.array([[1],
                       [2],
                       [3]])
    actual = mlr._add_x0(X_test)
    expected = np.array([[1, 1],
                         [1, 2],
                         [1, 3]])
    np.testing.assert_array_equal(actual, expected)


def test__add_x0_two_cols_multiple_samples():
    X_test = np.array([[1, 9],
                       [2, 7],
                       [3, 8]])
    actual = mlr._add_x0(X_test)
    expected = np.array([[1, 1, 9],
                         [1, 2, 7],
                         [1, 3, 8]])
    np.testing.assert_array_equal(actual, expected)


# def test__calc_inner():
#     X_aug = np.array([[1, 5, 1, 1],
#                       [1, 1, 1, 1],
#                       [1, 1, 2, 3]])
#     w = np.array([2, 4, 5, 6])
#
#     expected = np.array([33, 17, 34])
#     actual = mlr._calc_inner(X_aug, w)
#     np.testing.assert_array_equal(actual, expected)
#
#
# def test__calculate_inner_sum():
#     W = np.array([[1.53822067, -0.94362558, -0.16267383],
#                   [-1.43154505, 1.21260058, -0.41336865],
#                   [-1.14887306, -0.88885461, 0.49175771]])
#
#     X_test_aug = np.array([[1, 0, 1],
#                            [1, 5, 0],
#                            [1, 3, 10.25],
#                            [1, 0, 5]])
#
#     k = 0
#
#     expected = np.exp(np.array([1.37554684, -3.17990723, -2.96006283, 0.72485152]))
#     actual = mlr._calculate_inner_sum(X_test_aug, W, k)
#     np.testing.assert_allclose(actual, expected, atol=1e-16)


def test__calculate_inner_sums():
    X_test_aug = np.array([[1, 0, 1],
                           [1, 5, 0],
                           [1, 3, 10.25],
                           [1, 0, 5]])

    W = np.array([[1.53822067, -0.94362558, -0.16267383],
                  [-1.43154505, 1.21260058, -0.41336865],
                  [-1.14887306, -0.88885461, 0.49175771]])

    # calculate expected values by performing summations separately
    # see Note 1 in gradient_descent.py for explanation of why matrix multiplication can be used.
    expected = np.ones((len(W), len(X_test_aug)))
    for k in range(len(W)):
        w = W[k]
        expected[k] = np.exp(np.matmul(X_test_aug, w))
    actual = mlr._calculate_inner_sums(X_test_aug, W)
    np.testing.assert_allclose(actual, expected, atol=1e-16)


def test__calculate_outer_sum():
    inner_sums = np.array([[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12]])
    R_sums = inner_sums[1]

    actual = mlr._calculate_outer_sum(inner_sums, R_sums)
    expected = np.array([11, 13, 15, 17])
    np.testing.assert_array_equal(actual, expected)


def test__calc_conditional_proba__R0():
    inner_sums = np.array([[1.37554684, -3.17990723, -2.96006283, 0.72485152],
                           [-1.8449137, 4.63145785, -2.03077197, -3.4983883],
                           [-0.65711535, -5.59314611, 1.22507964, 1.30991549]])

    inner_sums = np.exp(inner_sums)
    R = 0
    ones = np.ones((4,))
    bottom = np.sum(inner_sums, axis=0) - inner_sums[R] + ones
    expected = 1 / bottom

    # actual results
    actual = mlr._calc_conditional_proba_R(inner_sums, R)
    np.testing.assert_allclose(actual, expected, atol=1e-16)


def test__calc_conditional_proba__R1():
    inner_sums = np.array([[1.37554684, -3.17990723, -2.96006283, 0.72485152],
                           [-1.8449137, 4.63145785, -2.03077197, -3.4983883],
                           [-0.65711535, -5.59314611, 1.22507964, 1.30991549]])

    inner_sums = np.exp(inner_sums)
    R = 1
    ones = np.ones((4,))
    bottom = np.sum(inner_sums, axis=0) - inner_sums[R] + ones
    expected = 1 / bottom

    # actual results
    actual = mlr._calc_conditional_proba_R(inner_sums, R)
    np.testing.assert_allclose(actual, expected, atol=1e-16)


def test__calc_conditional_proba__R2():
    inner_sums = np.array([[1.37554684, -3.17990723, -2.96006283, 0.72485152],
                           [-1.8449137, 4.63145785, -2.03077197, -3.4983883],
                           [-0.65711535, -5.59314611, 1.22507964, 1.30991549]])

    inner_sums = np.exp(inner_sums)
    R = 2
    ones = np.ones((4,))
    bottom = np.sum(inner_sums, axis=0) - inner_sums[R] + ones
    expected = 1 / bottom

    # actual results
    actual = mlr._calc_conditional_proba_R(inner_sums, R)
    np.testing.assert_allclose(actual, expected, atol=1e-16)


def test__calc_all_conditional_proba():
    X_test_aug = np.array([[1, 0, 1],
                           [1, 5, 0],
                           [1, 3, 10.25],
                           [1, 0, 5]])

    W = np.array([[1.53822067, -0.94362558, -0.16267383],
                  [-1.43154505, 1.21260058, -0.41336865],
                  [-1.14887306, -0.88885461, 0.49175771]])

    # calculate expected values
    inner_sums = np.array([[1.37554684, -3.17990723, -2.96006283, 0.72485152],
                           [-1.8449137, 4.63145785, -2.03077197, -3.4983883],
                           [-0.65711535, -5.59314611, 1.22507964, 1.30991549]])

    inner_sums = np.exp(inner_sums)
    expected_T = np.zeros((3, 4))
    for R in range(3):
        ones = np.ones((4,))
        bottom = np.sum(inner_sums, axis=0) - inner_sums[R] + ones
        expected_T[R] = 1 / bottom
    expected = expected_T.T
    actual = mlr._calc_all_conditional_proba(X_test_aug, W)
    np.testing.assert_allclose(actual, expected, atol=1e-16)


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


def test__get_largest_proba():
    y_pred_proba = np.array([[0.61204123, 0.18738013, 0.20057865],
                             [0.00988405, 0.98023551, 0.00988044],
                             [0.17089045, 0.17393603, 0.65517352],
                             [0.30960227, 0.21658015, 0.47381757]])

    expected = np.array([0, 1, 2, 2])
    actual = mlr._get_largest_proba(y_pred_proba)
    np.testing.assert_array_equal(actual, expected)


def test__init__():
    eta = 1
    epsilon = 2
    model = mlr.MulticlassLogisticRegression(eta, epsilon)
    assert model.eta == eta
    assert model.epsilon == epsilon
    assert model.weights is None


def test_fit():
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
    model = mlr.MulticlassLogisticRegression(eta=0.01, epsilon=0.01)
    model.fit(X, y)
    assert len(model.weights) == 3


def test_predict_proba():
    X_test = np.array([[0, 1],
                       [5, 0],
                       [3, 10.25],
                       [0, 5]])

    W = np.array([[1.53822067, -0.94362558, -0.16267383],
                  [-1.43154505, 1.21260058, -0.41336865],
                  [-1.14887306, -0.88885461, 0.49175771]])

    # calculate expected values
    inner_sums = np.array([[1.37554684, -3.17990723, -2.96006283, 0.72485152],
                           [-1.8449137, 4.63145785, -2.03077197, -3.4983883],
                           [-0.65711535, -5.59314611, 1.22507964, 1.30991549]])

    inner_sums = np.exp(inner_sums)
    expected_T = np.zeros((3, 4))
    for R in range(3):
        ones = np.ones((4,))
        bottom = np.sum(inner_sums, axis=0) - inner_sums[R] + ones
        expected_T[R] = 1 / bottom
    expected = expected_T.T

    sums = np.sum(expected, axis=1).reshape(-1, 1)
    standardized = expected / sums

    # set up model
    model = mlr.MulticlassLogisticRegression(eta=0.01, epsilon=0.01)
    model.weights = W

    # get results
    actual = model.predict_proba(X_test)
    np.testing.assert_allclose(actual, standardized, atol=1e-16)


def test_predict():
    X_test = np.array([[0, 1],
                       [5, 0],
                       [3, 10.25],
                       [0, 5]])

    W = np.array([[1.53822067, -0.94362558, -0.16267383],
                  [-1.43154505, 1.21260058, -0.41336865],
                  [-1.14887306, -0.88885461, 0.49175771]])

    # set up model
    model = mlr.MulticlassLogisticRegression(eta=0.01, epsilon=0.01)
    model.weights = W

    expected = np.array([0,1,2,2])
    actual = model.predict(X_test)
    np.testing.assert_array_equal(actual, expected)
