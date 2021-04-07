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
    assert model.classifiers is None


def test_fit():
    X = np.array([[1, 1],
                  [0, 0],
                  [1, 0],
                  [5, 1],
                  [5, 2],
                  [5, -1],
                  [3, 10],
                  [3, 10.5],
                  [3, 11]])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    model = mlr.MulticlassLogisticRegression(eta=0.01, epsilon=0.5)

    model.fit(X, y)
    assert len(model.classifiers) == 3
