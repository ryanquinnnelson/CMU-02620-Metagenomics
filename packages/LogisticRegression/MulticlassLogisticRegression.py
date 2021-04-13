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
def _standardize_probabilities(y_pred_proba):
    """

    :param y_pred_proba:
    :return:
    """

    sums = np.sum(y_pred_proba, axis=1).reshape(-1, 1)
    standardized = y_pred_proba / sums
    return standardized


def _update_predictions(y_pred, y_pred_proba_highest, y_pred_proba_k, k):
    """
    Updates label predictions by comparing current highest probabilities for each sample
    with the predicted probabilities for the kth class.

    Updates highest probabilities for any samples if kth class prediction probabilities are higher.

    :param y_pred:
    :param y_pred_proba_highest:
    :param y_pred_proba_k:
    :param k:
    :return:
    """
    # determine if any probabilities for kth class are higher than the current highest probabilities
    diff = y_pred_proba_k - y_pred_proba_highest

    # get indexes of all samples where kth class probability was higher
    idx = np.argwhere(diff > 0)

    # update those indexes in y_pred to be equal to k
    y_pred[idx] = k

    # update y_pred_proba_highest with higher probabilities from kth class
    y_pred_proba_highest[idx] = y_pred_proba_k[idx]


class MulticlassLogisticRegression2:
    """
    Implements multiclass logistic regression.

    This version stores the binary Logistic Regression classifiers fit to each class,
    and it uses the classifiers directly for predictions.
    """

    # tested
    def __init__(self, eta, epsilon, penalty=None, l2_lambda=0, max_iter=100):
        """

        :param eta:
        :param epsilon:
        :param penalty:
        :param l2_lambda:
        :param max_iter:
        """
        self.eta = eta
        self.epsilon = epsilon
        self.classifiers = None
        self.penalty = penalty
        self.l2_lambda = l2_lambda
        self.max_iter = max_iter

    # tested, sparse-enabled
    def fit(self, X, y):
        """
        Todo - add ability to turn off verbose printing.

        :param X: Assumes X is not augmented.
        :param y:
        :return:
        """
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

    # tested, sparse-enabled
    def predict_proba(self, X):
        """
        Calculates probabilities for each class for each sample.

        :param X: Assumes X is not augmented.
        :return: N x K matrix
        """
        K = len(self.classifiers)
        N = X.shape[0]
        y_pred_proba_T = np.zeros((K, N))  # transposed to make row replacement easier

        for k, classifier in enumerate(self.classifiers):
            predict_proba_k = classifier.predict_proba(X)[:, 1]  # probability of 1 for this class
            y_pred_proba_T[k] = predict_proba_k

        y_pred_proba = y_pred_proba_T.T
        y_pred_proba_standardized = _standardize_probabilities(y_pred_proba)
        return y_pred_proba_standardized

    # tested, sparse-enabled
    def predict(self, X):
        """
        Selects the class which has the highest probability for each sample.
        Note: Maximum number of allowed classes is 127.

        :param X: Assumes X is not augmented.
        :return:
        """
        N = X.shape[0]
        y_pred_proba_highest = np.zeros((N,))  # contains highest probabilities for each sample so far
        y_pred = np.zeros((N,), dtype=np.int8)  # doesn't allow more than 127 classes

        for k, classifier in enumerate(self.classifiers):
            predict_proba_k = classifier.predict_proba(X)[:, 1]  # probability of 1 for this class
            _update_predictions(y_pred, y_pred_proba_highest, predict_proba_k, k)

        return y_pred
