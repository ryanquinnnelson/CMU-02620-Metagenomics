import numpy as np
import packages.LogisticRegression.gradient_descent as gd


# tested
def _set_weights(X):
    """
    Creates an array of weights with each element set to 0.
    :param cols:
    :return:
    """
    cols = X.shape[1]
    return np.zeros(cols)


# tested
# does not work properly with a single sample
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


class LogisticRegression:

    # tested
    def __init__(self, eta, epsilon, penalty=None, l2_lambda=0):
        """

        :param eta: learning rate
        :param epsilon: convergence threshold
        """
        self.eta = eta
        self.epsilon = epsilon
        self.weights = None
        self.penalty = penalty
        self.l2_lambda = l2_lambda

    def fit(self, X, y):
        """

        :param X: L x n matrix, where L is the number of samples and n is the number of features
        :param y: L x 1 matrix
        :return:
        """
        # append imaginary column X_0=1 to accommodate w_0
        X_aug = _add_x0(X)

        # set initial weights
        weights = _set_weights(X_aug)

        # perform gradient descent until convergence
        weights = gd.gradient_descent(X_aug, y, weights, self.eta, self.epsilon, self.penalty, self.l2_lambda)
        self.weights = weights

        return self

    # tested
    def predict(self, X):
        """
        Returns predicted label for each sample.
        :param X: L x n matrix, where L is the number of samples and n is the number of features
        :return: L x 1 vector
        """
        # append imaginary column X_0=1 to accommodate w_0
        X_aug = _add_x0(X)

        y_pred = gd.get_y_predictions(X_aug, self.weights)
        return np.round(y_pred)

    def predict_proba(self, X):
        """
        Probability estimates. Returned estimates for all classes are ordered by the label of classes.
        Note: Currently implemented for data with two classes.

        :param X: L x n matrix, where L is the number of samples and n is the number of features
        :return: L x j vector, where j is the number of classes
        """
        # append imaginary column X_0=1 to accommodate w_0
        X_aug = _add_x0(X)

        # predictions for Y=1
        y_pred_1 = gd.get_y_predictions(X_aug, self.weights)

        # predictions for Y=0
        rows = y_pred_1.shape[0]
        y_pred_0 = np.ones(rows) - y_pred_1

        return np.column_stack((y_pred_0, y_pred_1))
