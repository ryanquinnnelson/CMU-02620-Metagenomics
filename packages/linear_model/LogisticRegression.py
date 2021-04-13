"""
Implements Logistic Regression as defined by Machine Learning (Mitchell).

Implementation augments the data for predictions to accommodate w0 term in the calculations. Data is augmented by
 adding a column of ones as the first column.
"""

import numpy as np
import packages.linear_model.gradient_descent as gd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack


# tested
def _set_weights(X):
    """
    Creates an array of weights with each element set to 0.

    :param X: L x J matrix, where L is the number of samples and J is the number of features.
            Assumes X is augmented for w0 already.
    :return: J x 1 array
    """
    cols = X.shape[1]
    return np.zeros(cols)


# tested
def _add_x0(X):
    """
    Adds a column to the left of matrix X with each element set to 1.
    Todo - make this function work for a single sample


    :param X: L x J matrix, where L is the number of samples and J is the number of features.
            Assumes X is not augmented for w0 yet.

    :return: L x (J+1) matrix
    """
    X_sparse = csr_matrix(X)  # convert to sparse matrix if not already sparse
    rows = X_sparse.shape[0]
    ones = np.ones(rows)

    # source: https://stackoverflow.com/questions/41937786/add-column-to-a-sparse-matrix
    X_aug = hstack((ones[:, None], X_sparse))
    return X_aug


class LogisticRegression:
    """
    Implements Logistic Regression for two-class (binary) data.
    """

    # tested
    def __init__(self, eta, epsilon, penalty=None, l2_lambda=0, max_iter=100):
        """
        Initializes instance of the class.

        :param eta: float, learning rate
        :param epsilon: float, convergence threshold
        :param penalty: str, penalty type to use. Default is None. Current implementation allows 'l2'.
        :param l2_lambda: float, value of l2 penalty if that penalty is used. Default is 0.
        :param max_iter: int, number of iterations allowed during convergence. Exceeding this number stops the algorithm
                and returns the current weights at that point. Default is 100.
        """
        self.eta = eta
        self.epsilon = epsilon
        self.weights = None
        self.penalty = penalty
        self.l2_lambda = l2_lambda
        self.max_iter = max_iter

    def fit(self, X, y):
        """
        Estimates the feature weights using the data.

        :param X: L x J matrix, where L is the number of samples and J is the number of dimensions in a sample
        :param y: L x 1 matrix, labels for each sample
        :return:
        """
        # append imaginary column X_0=1 to accommodate w_0
        X_aug = _add_x0(X)

        # convert to sparse matrix
        X_aug_sparse = csr_matrix(X_aug)

        # set initial weights
        weights = _set_weights(X_aug_sparse)

        # perform gradient descent until convergence
        weights = gd.gradient_descent(X_aug_sparse, y, weights, self.eta, self.epsilon,
                                      self.penalty, self.l2_lambda, self.max_iter)

        # save weights in this instance
        self.weights = weights

        return self

    # tested
    def predict(self, X):
        """
        Get predicted label for each sample.

        :param X: L x J matrix, where L is the number of samples and J is the number of dimensions in a sample
        :return: L x 1 vector
        """
        # append imaginary column X_0=1 to accommodate w_0
        X_aug = _add_x0(X)

        # convert to sparse matrix
        X_aug_sparse = csr_matrix(X_aug)

        # get prediction probabilities
        y_pred_proba = gd.get_y_predictions(X_aug_sparse, self.weights)

        # round to nearest whole value
        y_pred = np.round(y_pred_proba)

        return y_pred

    def predict_proba(self, X):
        """
        Get probability estimates for classes. Estimates for all classes are ordered by class label
        (i.e. [0.2,0.8] indicates 20% probability of class 0, 80% probability of class 1).

        :param X: L x J matrix, where L is the number of samples and J is the number of dimensions in a sample
        :return: L x C vector, where C is the number of classes
        """
        # append imaginary column X_0=1 to accommodate w_0
        X_aug = _add_x0(X)

        # convert to sparse matrix
        X_aug_sparse = csr_matrix(X_aug)

        # predictions for Y=1
        y1_pred_proba = gd.get_y_predictions(X_aug_sparse, self.weights)

        # predictions for Y=0
        rows = y1_pred_proba.shape[0]
        y0_pred_proba = np.ones(rows) - y1_pred_proba

        # combine predictions for each class into a single matrix
        y_pred_proba = np.column_stack((y0_pred_proba, y1_pred_proba))

        return y_pred_proba
