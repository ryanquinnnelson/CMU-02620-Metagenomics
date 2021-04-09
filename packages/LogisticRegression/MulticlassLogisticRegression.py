"""
One-vs-all is a strategy that involves training N distinct binary classifiers,
 each designed to recognize a specific class. After that we collectively use those N classifiers to predict
 the correct class.

 We augment the data for predictions to accommodate w0 in the calculations. We don't augmented the data for fitting the
 model because logistic regression already augments the data.
"""
from packages.LogisticRegression.LogisticRegression import LogisticRegression
import numpy as np
import copy


# tested
def _calculate_number_classes(y):
    """
    Determines the number of classes among the labels.

    :param y: n x 1 array, where n is the number of samples. Represents labels matching to samples.
    :return: int, number of classes
    """
    return len(np.unique(y))


# tested
def _convert_to_binary_classes(y, label):
    """
    Changes labels of samples with class=label to one and labels of all other samples to zero.

    :param y: n x 1 array, where n is the number of samples. Represents labels matching to samples.
    :param label: int, represents label of the class to be defined with ones.
    :return: n x 1 array of binary values
    """
    y_binary = copy.deepcopy(y)  # avoid modifying original array
    y_binary[y == label] = 1  # set class which remains as 1
    y_binary[y != label] = 0  # set other classes to zero
    return y_binary


# tested
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


# # tested
# def _calc_inner(X, w):
#     """
#     Performs the inner calculation w_0 + SUM_i w_i X_i^L. See Note 1 in gradient_descent.py for explanation of function logic.
#
#     :param X:  L x n matrix, where L is the number of samples and n is the number of features
#     :param w: n x 1 vector
#     :return: L x 1 vector
#     """
#     return np.matmul(X, w)
#
#
# # tested
# def _calculate_inner_sum(X, W, k):
#     """
#
#     :param X: augmented data
#     :param W:
#     :param k:
#     :return:
#     """
#     w = W[k]
#     Xw = _calc_inner(X, w)
#     Xw_exp = np.exp(Xw)
#     return Xw_exp


# tested
def _calculate_outer_sum(inner_sums, R_sums):
    """
    It is more efficient to subtract one row from another row than to create a second matrix with R-1 rows.

    :param inner_sums:
    :return:
    """

    # sum K-1 classes
    R_minus_one_sums = np.sum(inner_sums, axis=0) - R_sums

    # add 1 to each sample value
    n_samples = inner_sums.shape[1]
    ones = np.ones((n_samples,))
    bottom = ones + R_minus_one_sums

    return bottom


# tested
def _calculate_inner_sums(X, W):
    """
    Calculates all inner sums at the same time. After performing each class inner summation separately per the method
    in Note 1 gradient_descent.py, I found that it was possible to get the same results by modifying the order and
    terms of the matrix multiplication.

    :param X:
    :param W:
    :return:
    """
    summations = np.matmul(W, X.T)
    inner_sums = np.exp(summations)
    return inner_sums


# tested
def _calc_conditional_proba_R(inner_sums, R):
    """

    P(Y=R|X^L) = 1 / a

    where
    - a = 1 + SUM_k=1^R-1 exp(w_k0 + SUM_i=1^n w_ki X_i^L)
    - R: the Rth class
    - n: number of samples

    :param inner_sums:
    :param R:
    :return:
    """
    n_classes = len(inner_sums)
    n_samples = inner_sums.shape[1]

    # get sums for class R so they can be subtracted from total
    R_sums = inner_sums[R]

    # sum up the bottom terms without sums from class R
    bottom = _calculate_outer_sum(inner_sums, R_sums)

    y_pred = 1 / bottom
    return y_pred


# tested
def _calc_all_conditional_proba(X, W):
    """

    :param X: augmented data
    :param W:
    :return:
    """

    N = len(X)  # number of samples
    K = len(W)  # number of classes

    # calculate inner sums for reuse
    inner_sums = _calculate_inner_sums(X, W)

    predictions = np.zeros((K, N))  # transposed for ease of row replacement
    for k in range(K):
        proba_k = _calc_conditional_proba_R(inner_sums, k)
        predictions[k] = proba_k

    return predictions.T


# tested
def _standardize_probabilities(y_pred_proba):
    """

    :param y_pred_proba:
    :return:
    """

    sums = np.sum(y_pred_proba, axis=1).reshape(-1, 1)
    standardized = y_pred_proba / sums
    return standardized


# tested
def _get_largest_proba(y_predict_proba):
    return np.argmax(y_predict_proba, axis=1)


class MulticlassLogisticRegression:
    """
    This version stores the regression coefficient weights from each of the binary Logistic Regression classifiers fit
    to each class, and it uses the weights for predictions.
    """

    # tested
    def __init__(self, eta, epsilon):
        self.eta = eta
        self.epsilon = epsilon
        self.weights = None

    # tested
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
        Todo - Determine if standardization should really be necessary.

        :param X:
        :return: N x K matrix
        """
        # augment the data so w0 makes sense
        X_aug = _add_x0(X)

        # calculate probabilities per class
        y_pred_proba = _calc_all_conditional_proba(X_aug, self.weights)

        y_pred_proba_standardized = _standardize_probabilities(y_pred_proba)

        return y_pred_proba_standardized

    def predict(self, X):
        """
        Selects the class which has the highest probability for each sample.
        :param X:
        :return:
        """
        y_pred_proba_standardized = self.predict_proba(X)

        # for each sample choose the class with the highest probability
        y_pred = _get_largest_proba(y_pred_proba_standardized)
        return y_pred


class MulticlassLogisticRegression2:
    """
    This version stores the binary Logistic Regression classifiers fit to each class,
    and it uses the classifiers for predictions.
    """

    # tested
    def __init__(self, eta, epsilon, penalty=None, l2_lambda=0, max_iter=100):
        self.eta = eta
        self.epsilon = epsilon
        self.classifiers = None
        self.penalty = penalty
        self.l2_lambda = l2_lambda
        self.max_iter = max_iter

    # tested
    def fit(self, X, y):
        # determine how many binary classifiers must be trained
        n_classifiers = _calculate_number_classes(y)
        print('n_classifiers', n_classifiers)
        # train binary classifier for each class
        classifiers = []
        for i in range(n_classifiers):
            print('training classifier {}'.format(i))
            lr = LogisticRegression(eta=self.eta,
                                    epsilon=self.epsilon,
                                    penalty=self.penalty,
                                    l2_lambda=self.l2_lambda,
                                    max_iter=self.max_iter)

            # convert to binary classes
            y_binary = _convert_to_binary_classes(y, i)

            # fit binary classifier
            lr.fit(X, y_binary)

            # retain weights for this classifier
            classifiers.append(lr)

        self.classifiers = classifiers
        return self

    # tested
    def predict_proba(self, X):
        """
        Calculates probabilities for each class for each sample.

        :param X:
        :return: N x K matrix
        """
        K = len(self.classifiers)
        N = len(X)
        y_pred_proba_T = np.zeros((K, N))  # transposed to make row replacement easier

        for k, classifier in enumerate(self.classifiers):
            predict_proba_k = classifier.predict_proba(X)[:, 1]  # probability of 1 for this class
            y_pred_proba_T[k] = predict_proba_k

        y_pred_proba = y_pred_proba_T.T
        y_pred_proba_standardized = _standardize_probabilities(y_pred_proba)
        return y_pred_proba_standardized

    # tested
    def predict(self, X):
        """
        Selects the class which has the highest probability for each sample.
        :param X:
        :return:
        """
        K = len(self.classifiers)
        N = len(X)
        y_pred_proba_T = np.zeros((K, N))  # transposed to make row replacement easier

        for k, classifier in enumerate(self.classifiers):
            predict_proba_k = classifier.predict_proba(X)[:, 1]  # probability of 1 for this class
            y_pred_proba_T[k] = predict_proba_k

        y_pred_proba = y_pred_proba_T.T   # no need to standardize results

        # for each sample choose the class with the highest probability
        y_pred = _get_largest_proba(y_pred_proba)
        return y_pred
