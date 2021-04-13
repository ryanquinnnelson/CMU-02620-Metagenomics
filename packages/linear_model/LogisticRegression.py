import numpy as np
import packages.linear_model.gradient_descent as gd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack


# tested
def _set_weights(X):
    """
    Creates an array of weights with each element set to 0.

    :param X: N x J matrix, where N is the number of samples and J is the number of features.
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


    :param X: N x J matrix, where N is the number of samples and J is the number of features.
            Assumes X is not augmented for w0 yet.

    :return: N x (J+1) matrix
    """
    X_sparse = csr_matrix(X)  # convert to sparse matrix if not already sparse
    rows = X_sparse.shape[0]
    ones = np.ones(rows)

    # source: https://stackoverflow.com/questions/41937786/add-column-to-a-sparse-matrix
    X_aug = hstack((ones[:, None], X_sparse))
    return X_aug


class LogisticRegression:
    """
    Implements logistic regression using gradient descent as defined by Machine Learning (Mitchell).
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
        self.weights = None
        self.penalty = penalty
        self.l2_lambda = l2_lambda
        self.max_iter = max_iter

    def fit(self, X, y):
        """
        Fits the model to the data.

        :param X: L x n matrix, where L is the number of samples and n is the number of features
        :param y: L x 1 matrix
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

        # convert to sparse matrix
        X_aug_sparse = csr_matrix(X_aug)

        y_pred = gd.get_y_predictions(X_aug_sparse, self.weights)
        return np.round(y_pred)

    def predict_proba(self, X):
        """
        Probability estimates. Returned estimates for all classes are ordered by the label of classes.
        Implemented for binary data.

        :param X: L x n matrix, where L is the number of samples and n is the number of features
        :return: L x j vector, where j is the number of classes
        """
        # append imaginary column X_0=1 to accommodate w_0
        X_aug = _add_x0(X)

        # convert to sparse matrix
        X_aug_sparse = csr_matrix(X_aug)

        # predictions for Y=1
        y_pred_1 = gd.get_y_predictions(X_aug_sparse, self.weights)

        # predictions for Y=0
        rows = y_pred_1.shape[0]
        y_pred_0 = np.ones(rows) - y_pred_1

        return np.column_stack((y_pred_0, y_pred_1))
