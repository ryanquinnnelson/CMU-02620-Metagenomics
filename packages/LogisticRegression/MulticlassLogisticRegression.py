"""
One-vs-all is a strategy that involves training N distinct binary classifiers,
 each designed to recognize a specific class. After that we collectively use those N classifiers to predict
 the correct class.
"""
from packages.LogisticRegression.LogisticRegression import LogisticRegression
import numpy as np
import copy


# tested
def _calculate_number_classes(y):
    return len(np.unique(y))


def _add_x0(X):
    """
    Adds a column to the left of matrix X with each element set to 1.
    :param X:
    :param rows:
    :return:
    """
    rows = X.shape[0]
    ones = np.ones(rows)
    return np.insert(X, 0, ones, axis=1)


# tested
def _convert_to_binary_classes(y, i):
    y_binary = copy.deepcopy(y)  # avoid modifying original array
    y_binary[y == i] = 1  # set class which remains as 1
    y_binary[y != i] = 0  # set other classes to zero
    return y_binary


def _calc_inner(X, w):
    """
    Performs the inner calculation w_0 + SUM_i w_i X_i^L. See Note 1 in gradient descent.

    :param X: augmented data
    :param w:
    :return:
    """
    return np.matmul(X, w)


def _get_conditional_proba(X, W, R):
    """

    :param X: augmented data
    :param W:
    :param R:
    :return:
    """

    K = len(W)
    N = len(X)
    terms = np.zeros((K - 1, N))
    row = 0

    # consider classes not equal to R
    for k in range(K):

        if k != R:
            w = W[k]
            Xw = _calc_inner(X, w)
            Xw_exp = np.exp(Xw)
            terms[row] = Xw_exp
            row += 1

    # sum up the bottom terms
    sum_terms = np.sum(terms, axis=0)
    ones = np.ones((N, ))
    bottom = ones + sum_terms
    y_pred = 1 / bottom

    return y_pred


def _get_all_conditional_proba(X, W):
    """

    :param X: augmented data
    :param W:
    :return:
    """

    N = len(X)
    K = len(W)

    predictions = np.zeros((K, N))  # transposed for ease of row replacement

    for k in range(K):
        proba_k = _get_conditional_proba(X, W, k)
        predictions[k] = proba_k

    return predictions.T


def _standardize_probabilities(predictions):

    sums = np.sum(predictions, axis=1).reshape(-1, 1)
    standardized = predictions / sums
    return standardized


class MulticlassLogisticRegression:

    # tested
    def __init__(self, eta, epsilon):
        self.eta = eta
        self.epsilon = epsilon
        self.weights = None

    # tested - perform additional checks to make sure weights match expectations
    def fit(self, X, y):
        # determine how many binary classifiers must be trained
        n_classifiers = _calculate_number_classes(y)
        J = X.shape[1]

        # train binary classifier for each class
        weights = np.zeros((n_classifiers, J + 1))  # extra col for w0
        for i in range(n_classifiers):
            lr = LogisticRegression(eta=self.eta, epsilon=self.epsilon)

            # convert to binary classes
            y_binary = _convert_to_binary_classes(y, i)

            # fit binary classifier
            lr.fit(X, y_binary)

            # retain weights for this classifier
            weights[i] = lr.weights

        self.weights = weights
        return self

    def predict_proba(self, X):
        """
        Calculates probabilities for each class for each sample.
        :param X:
        :return: N x K matrix
        """
        # augment the data so w0 makes sense
        pass

    def predict(self, X):
        """
        Gets the class which has the highest probability.
        :param X:
        :return:
        """
        # augment the data so w0 makes sense
        pass
