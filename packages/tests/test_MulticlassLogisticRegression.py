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

    X_aug = np.array([[1, 1, 1],
                      [1, 0, 0],
                      [1, 1, 0],
                      [1, 0, 4],
                      [1, 5, 1],
                      [1, 5, 2],
                      [1, 5, -1],
                      [1, 5, 10],
                      [1, 3, 10],
                      [1, 3, 10.5],
                      [1, 3, 11]])
    actual = mlr._get_all_conditional_proba(X_aug, W)
    print(actual)
    standardized = mlr._standardize_probabilities(actual)
    print(standardized)
    1/0
    # [[0.3500007  0.39255209 0.37355716]
    #  [0.34503378 0.34230381 0.33687351]
    # [0.34147442
    # 0.3849923
    # 0.3340664]
    # [0.33180575 0.33893353 0.50640628]
    # [0.29150449
    # 0.57550672
    # 0.29791982]
    # [0.31368516 0.57792385 0.33294806]
    # [0.24443735
    # 0.55252551
    # 0.23461234]
    # [0.37235007 0.44443846 0.64519658]
    # [0.32407774
    # 0.35428839
    # 0.7006866]
    # [0.31574771 0.34218574 0.71763083]
    # [0.30702207
    # 0.33008027
    # 0.73396896]]
