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


# tested
def _convert_to_binary_classes(y, i):
    y_binary = copy.deepcopy(y)  # avoid modifying original array
    y_binary[y == i] = 1  # set class which remains as 1
    y_binary[y != i] = 0  # set other classes to zero
    return y_binary


class MulticlassLogisticRegression:

    def __init__(self, eta, epsilon):
        self.eta = eta
        self.epsilon = epsilon
        self.classifiers = None

    def fit(self, X, y):
        # determine how many binary classifiers must be trained
        n_classifiers = _calculate_number_classes(y)

        # train binary classifiers
        classifiers = []
        for i in range(n_classifiers):
            lr = LogisticRegression(self.eta, self.epsilon)

            # convert to binary classes
            y_binary = _convert_to_binary_classes(y, i)
            lr.fit(X, y_binary)

            # save trained classifier
            classifiers.append(lr)

        self.classifiers = classifiers
        return self

    def predict_proba(self, X):
        """
        Calculates probabilities for each class for each sample.
        :param X:
        :return: N x K matrix
        """
        pass

    def predict(self, X):
        """
        Gets the class which has the highest probability.
        :param X:
        :return:
        """
        pass
