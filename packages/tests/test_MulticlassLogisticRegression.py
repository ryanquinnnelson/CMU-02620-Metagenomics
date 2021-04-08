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
        expected[k] = np.matmul(X_test_aug, w)

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

#
# def test__calc_conditional_proba():
#     W = np.array([[1.53822067, -0.94362558, -0.16267383],
#                   [-1.43154505, 1.21260058, -0.41336865],
#                   [-1.14887306, -0.88885461, 0.49175771]])
#
#     inner_sums = np.array([[1.37554684, -3.17990723, -2.96006283, 0.72485152],
#                          [-1.8449137, 4.63145785, -2.03077197, -3.4983883],
#                          [-0.65711535, -5.59314611, 1.22507964, 1.30991549]])
#
#     inner_sums = np.exp(inner_sums)



    # actual = mlr._calc_conditional_proba(X_test_aug, W)





#
# def test__init__():
#     eta = 1
#     epsilon = 2
#     model = mlr.MulticlassLogisticRegression(eta, epsilon)
#     assert model.eta == eta
#     assert model.epsilon == epsilon
#     assert model.weights is None
#
#
# def test_fit():
#     X = np.array([[1, 1],
#                   [0, 0],
#                   [1, 0],
#                   [0, 4],
#                   [5, 1],
#                   [5, 2],
#                   [5, -1],
#                   [5, 10],
#                   [3, 10],
#                   [3, 10.5],
#                   [3, 11]])
#     y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
#     model = mlr.MulticlassLogisticRegression(eta=0.01, epsilon=0.01)
#     model.fit(X, y)
#     assert len(model.weights) == 3
#

#
# #
# # def test__get_all_conditional_proba():
# #     X = np.array([[1, 1],
# #                   [0, 0],
# #                   [1, 0],
# #                   [0, 4],
# #                   [5, 1],
# #                   [5, 2],
# #                   [5, -1],
# #                   [5, 10],
# #                   [3, 10],
# #                   [3, 10.5],
# #                   [3, 11]])
# #     y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
# #     model = mlr.MulticlassLogisticRegression(eta=0.01, epsilon=0.01)
# #     model.fit(X, y)
# #     W = model.weights
# #     print(W)
# #     print()
# #
# #     X_test_aug = np.array([[1, 0, 1],
# #                            [1, 5, 0],
# #                            [1, 3, 10.25],
# #                            [1, 0, 5]])
# #     actual = mlr._get_all_conditional_proba(X_test_aug, W)
# #     print(actual)
# #     print()
# #     standardized = mlr._standardize_probabilities(actual)
# #     print(standardized)
# #     print()
# #     1 / 0
