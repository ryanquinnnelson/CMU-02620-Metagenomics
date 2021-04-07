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
    model = mlr.MulticlassLogisticRegression(eta=0.01, epsilon=0.5)
    model.fit(X, y)
    assert len(model.weights) == 3


def test__calc_inner():
    X_aug = np.array([[1, 5, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 2, 3]])
    w = np.array([2, 4, 5, 6])

    expected = np.array([33, 17, 34])
    actual = mlr._calc_inner(X_aug, w)
    np.testing.assert_array_equal(actual, expected)


def test__get_conditional_proba__two_classes():
    X_aug = np.array([[1, 5, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 2, 3]])

    W = np.array([[0, 0, 0, 0],
                  [2, 4, 5, 6],
                  [1, 2, 3, 4]])

    actual = mlr._get_conditional_proba(X_aug, W, R=0)
    assert actual.shape == (3,)


def test__get_conditional_proba__three_classes():
    X_aug = np.array([[1, 5, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 2, 3]])

    W = np.array([[0, 0, 0, 0],
                  [2, 4, 5, 6],
                  [1, 2, 3, 4],
                  [5, 1, 2, 2]])

    actual = mlr._get_conditional_proba(X_aug, W, R=0)
    assert actual.shape == (3,)


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


def test__get_all_conditional_proba():
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
    model = mlr.MulticlassLogisticRegression(eta=0.01, epsilon=0.5)
    model.fit(X, y)
    W = model.weights
    print(W)
    print()

    X_aug = np.array([[1, 0, 1],
                      [1, 5, 0],
                      [1, 3, 10.25],
                      [1, 0, 5]])
    actual = mlr._get_all_conditional_proba(X_aug, W)
    print(actual)
    print()
    standardized = mlr._standardize_probabilities(actual)
    print(standardized)
    print()
    1 / 0

    # [[0.32373151 0.32512617 0.35114232]
    #  [0.24372189 0.51537179 0.24090632]
    #  [0.23229095 0.25281724 0.51489181]]
