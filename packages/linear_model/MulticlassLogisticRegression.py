"""
Implements Multiclass logistic regression using one-vs-all strategy.

One-vs-all involves training N distinct binary classifiers, each designed to recognize a specific class, then
 collectively use those N classifiers to predict the correct class.

 Implementation augments the data for predictions to accommodate w0 term in the calculations. Data is augmented by
 adding a column of ones as the first column.  Note that data is not augmented for fitting the model because
 LogisticRegression already augments the data for this.
"""
from packages.linear_model.LogisticRegression import LogisticRegression
import numpy as np
import copy


# tested
def _calculate_number_classes(y):
    """
    Determines the number of classes among the labels.

    :param y: L x 1 array, where L is the number of samples.
    :return: int, number of classes
    """
    return len(np.unique(y))


# tested
def _convert_to_binary_classes(y, label):
    """
    Changes labels of samples with class=label to one and labels of all other samples to zero.

    :param y: L x 1 array, where L is the number of samples.
    :param label: int, represents label of the class to be defined with ones.
    :return: L x 1 array of binary values
    """
    y_binary = copy.deepcopy(y)  # avoid modifying original array
    y_binary[y == label] = 1  # set class which remains as 1
    y_binary[y != label] = 0  # set other classes to zero
    return y_binary


# tested
def _standardize_probabilities(y_pred_proba):
    """
    Divides probabilities for a given sample by the total of all probabilities so that total sums to 1.0.

    :param y_pred_proba: L x K array, where L is the number of samples and K is the number of classes.
    :return: L x K array
    """

    sums = np.sum(y_pred_proba, axis=1).reshape(-1, 1)
    standardized = y_pred_proba / sums
    return standardized


def _update_predictions(y_pred, y_pred_proba_highest, y_pred_proba_k, k):
    """
    Efficiently updates label predictions by comparing current highest probabilities for each sample
    with the predicted probabilities for the kth class. Sets the predicted class to be k for samples where the kth
    class has a higher probability than the current highest probability.

    Updates highest probabilities for any samples if kth class prediction probabilities are higher.

    :param y_pred: L x 1 array, where L is the number of samples. Represents current class predictions for samples.
    :param y_pred_proba_highest: L x 1 array. Contains the highest probability found so far for each sample from
                                among the classes.
    :param y_pred_proba_k: L x 1 array. Contains prediction probabilities for the kth class.
    :param k: int, the kth class.
    :return: None
    """
    # determine if any probabilities for kth class are higher than the current highest probabilities
    diff = y_pred_proba_k - y_pred_proba_highest

    # get indexes of all samples where kth class probability was higher
    idx = np.argwhere(diff > 0)

    # update those indexes in y_pred to be equal to k
    y_pred[idx] = k

    # update y_pred_proba_highest with higher probabilities from kth class
    y_pred_proba_highest[idx] = y_pred_proba_k[idx]


class MulticlassLogisticRegression:
    """
    Implements multiclass logistic regression.

    This version stores the binary Logistic Regression classifiers fit to each class,
    and it uses the classifiers directly for predictions.
    """

    # tested
    def __init__(self, eta, epsilon, penalty=None, l2_lambda=0, max_iter=100, verbose=False):
        """
        Initializes an instance.

        :param eta: float, learning rate
        :param epsilon: float, convergence threshold
        :param penalty: str, penalty type to use. Default is None. Current implementation allows 'l2'.
        :param l2_lambda: float, value of l2 penalty if that penalty is used. Default is 0.
        :param max_iter: int, number of iterations allowed during convergence. Exceeding this number stops the algorithm
                and returns the current weights at that point. Default is 100.
        :param verbose: boolean, print progress updates to the console if True. Default is False.
        """
        self.eta = eta
        self.epsilon = epsilon
        self.classifiers = None
        self.penalty = penalty
        self.l2_lambda = l2_lambda
        self.max_iter = max_iter
        self.verbose = verbose

    # tested, sparse-enabled
    def fit(self, X, y):
        """
        Estimates the feature weights using the data.

        :param X: L x J matrix, where L is the number of samples and J is the number of dimensions in a sample.
                    Assumes X is not augmented.
        :param y: L x 1 array, class labels for each sample
        :return:
        """
        # determine how many binary classifiers must be trained
        n_classifiers = _calculate_number_classes(y)
        if self.verbose:
            print('n_classifiers', n_classifiers)

        # train binary classifier for each class
        classifiers = []
        for k in range(n_classifiers):
            if self.verbose:
                print('training classifier {}'.format(k))

            # train classifier for kth class
            lr = LogisticRegression(eta=self.eta,
                                    epsilon=self.epsilon,
                                    penalty=self.penalty,
                                    l2_lambda=self.l2_lambda,
                                    max_iter=self.max_iter)

            # convert to binary classes
            y_binary = _convert_to_binary_classes(y, k)

            # fit binary classifier
            lr.fit(X, y_binary)

            # retain classifier for predictions
            classifiers.append(lr)

        # save to instance
        self.classifiers = classifiers

        return self

    # tested, sparse-enabled
    def predict_proba(self, X):
        """
        Calculates probabilities for each class for each sample.

        :param X: L x J matrix, where L is the number of samples and J is the number of dimensions in a sample.
                    Assumes X is not augmented.
        :return: L x K matrix, where K is the number of classes.
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
        Predicts the class which has the highest probability for each sample.
        Note: Maximum number of allowed classes is 127.

        :param X: L x J matrix, where L is the number of samples and J is the number of dimensions in a sample.
                    Assumes X is not augmented.
        :return: L x 1 array
        """
        N = X.shape[0]
        y_pred_proba_highest = np.zeros((N,))  # contains highest probabilities for each sample so far
        y_pred = np.zeros((N,), dtype=np.int8)  # doesn't allow more than 127 classes

        for k, classifier in enumerate(self.classifiers):
            predict_proba_k = classifier.predict_proba(X)[:, 1]  # probability of 1 for this class
            _update_predictions(y_pred, y_pred_proba_highest, predict_proba_k, k)

        return y_pred
