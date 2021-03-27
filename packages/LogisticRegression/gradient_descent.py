"""
Implementation of gradient descent for Logistic Regression, as defined by Machine Learning (Mitchell).
Assumes "imaginary" X_0 = 1 for all samples has been added to the matrix to accommodate w_0 (i.e. X has been augmented
so that its first column is 1's).
"""

import numpy as np

"""
Note 1 - Explanation of _calc_inner()

Many of the formulas for Logistic Regression involve a calculation between n x 1 weights vector w and L x n feature 
matrix X:

w_0 + SUM_i^n w_i X_i^L
where
- i is the ith feature
- n is the number of features
- L is the number of samples

This can be accomplished efficiently using matrix multiplication, as explained below. As discussed in Mitchell, we 
augment X with imaginary X_0 column to accommodate w_0.

CASE 1 - Single feature, single sample
-----------------------------------------
X        w                 X_aug        w
|x11|    | w0 |            | 1 |x11|    | w0 | 
         | w1 |   ->                    | w1 |      
                      
We perform the summation using dot product and get a scalar.

X_aug dot w = 1*w0 + x11*w1


CASE 2 - Multiple features, single sample
-----------------------------------------
X            w                 X_aug            w
|x11|x12|    | w0 |            | 1 |x11|x12|    | w0 | 
             | w1 |   ->                        | w1 | 
             | w2 |                             | w2 |                             

                      
We perform the summation using dot product and get a scalar.

X_aug dot w = 1*w0 + x11*w1 + x12*w2


CASE 3 - Single feature, multiple samples
-----------------------------------------
X        w                 X_aug         w   
|x11|    | w0 |            | 1 |x11|     | w0 |
|X21|    | w1 |   ->       | 1 |x21|     | w1 |
|x31|                      | 1 |x31|

We could perform the summation for each sample separately and place the results into a vector, or we could use matrix 
multiplication and do all calculations simultaneously. The result is a vector.

               |  1*w0 + x11*w1  |
(X_aug)(w) =   |  1*w0 + x21*w1  |
               |  1*w0 + x31*w1  |


CASE 4 - Multiple features, multiple samples
--------------------------------------------
X                w             X_aug                w
|x11|x12|x13|    | w0 |        | 1 |x11|x12|x13|    | w0 |
|X21|x22|x23|    | w1 |   ->   | 1 |x21|x22|x23|    | w1 |
|x31|x32|x33|    | w2 |        | 1 |x31|x32|x33|    | w2 |
                 | w3 |                             | w3 | 

This works the same as CASE 3. The result is a vector.

               |  1*w0 + x11*w1 + x12*w2 + x13*w3  |
(X_aug)(w) =   |  1*w0 + x21*w1 + x22*w2 + x23*w3  |
               |  1*w0 + x31*w1 + x32*w2 + x33*w3  |

"""

"""
Note 2 - Explanation of _calc_gradient()
According to Mitchell, the ith partial in the gradient can be calculated as:

    d l(W)/d w_i = SUM_L X_i^L y_err^L
    
where 
- L is the number of samples
- i is the ith feature
- y_err = ( Y^L - P(Y^L = 1 |X^L,W))

This can be accomplished efficiently using matrix multiplication, as explained below. As discussed in Mitchell, we 
augment X with imaginary X_0 column to accommodate w_0.


CASE 1 - Single feature, single sample
-----------------------------------------
X         y_true     y_pred           X_aug         w
|x11|     | yt11 |      | yp11 |      | 1 |x11|    |w0|
                                                   |w1|
                                                   
y_err = | ye11 | = | yt11 - yp11 |

We could calculate each partial separately or we could calculate the gradient by transposing X and using 
matrix multiplication. The result is a vector.

l(W) / d w_0 = X_0 * y_err =   |   1*ye11 |

l(W) / d w_1 = X_1 * y_err =   | x11*ye11 |

gradient = (X_aug)^T(y_err) = | 1 |  x  | ye11| = |   1*ye11 |
                              |x11|               | x11*ye11 |

CASE 2 - Multiple features, single sample
-----------------------------------------
We'll simplify by assuming y_err has already been calculated.

X                y_err       X_aug                  w
|x11|x12|x13|    | y1 |      | 1 |x11|x12|x13|      |w0|
                                                    |w1|
                                                    |w2|
                                                    |w3|

gradient = (X_aug)^T(y_err) = | 1 |  x  | y1 |  =  |  1*y1|
                              |x11|                |x11*y1|
                              |x12|                |x12*y1|
                              |x13|                |x13*y1| 
                              
                              
CASE 3 - Single feature, multiple samples
-----------------------------------------
We'll simplify by assuming y_err has already been calculated.  

X       y_err     X_aug        w
|x11|   | y1 |    | 1 |x11|    |w0|
|x21|   | y2 |    | 1 |x21|    |w1|
|x31|   | y3 |    | 1 |x31|

                 X_0         y_err
                 | 1 |       | y1 | 
d l(W) / d w_0 = | 1 |  dot  | y2 |  = ( 1 *y1) + ( 1 *y2) + ( 1 *y3)
                 | 1 |       | y3 |


                 X_1         y_err
                 |x11|       | y1 | 
d l(W) / d w_1 = |x21|  dot  | y2 |  = (x11*y1) + (x21*y2) + (x31*y3)
                 |x31|       | y3 |



gradient = (X_aug)^T(y_err) =  | 1 | 1 | 1 |      |y1|     |  1*y1 +   1*y2 +   1*y3 |
                               |x11|x21|x31|   x  |y2|  =  |x11*y1 + x21*y2 + x31*y3 |
                                                  |y3|
  
CASE 3 - Multiple features, multiple samples
--------------------------------------------
We'll simplify by assuming y_err has already been calculated.  

X                y_err      X_aug                w
|x11|x12|x13|    | y1 |     | 1 |x11|x12|x13|    |w0|
|X21|x22|x23|    | y2 |     | 1 |X21|x22|x23|    |w1|
|x31|x32|x33|    | y3 |     | 1 |x31|x32|x33|    |w2|
                                                 |w3|

                            
                                | d w0 |    | ( 1 *y1) + ( 1 *y2) + ( 1 *y3) |
                                | d w1 |    | (x11*y1) + (x21*y2) + (x31*y3) |
gradient = (X_aug)^T(y_err) =   | d w2 | =  | (x12*y1) + (x22*y2) + (x32*y3) |
                                | d w3 |    | (x13*y1) + (x23*y2) + (x33*y3) |
"""

"""
Note 3 - Explanation of log likelihood calculation

According to Mitchell, log likelihood l(W) can be calculated as follows:

l(W) = SUM_L [ Y^L * A - ln( 1 + exp(A)) ]

where 
- L is the number of samples
- A = w_0 + SUM_i^n w_i X_i^L
- i is the ith feature
- n is the number of features

The math required to calculate A is found in Note 1. We can distribute the summation to get:

l(W) = SUM_L [ Y^L * A ] - SUM_L [ ln( 1 + exp(A)) ]
"""


# tested
def _update_weights(w, eta, gradient):
    """
    Updates regression coefficients using the following formula:
    W <- W + (eta * gradient)

    :param w: n x 1 vector
    :param eta:
    :param gradient: n x 1 vector
    :return:
    """
    change = eta * gradient
    return w + change


# tested
def _calc_inner(X, w):
    """
    Performs the inner calculation w_0 + SUM_i w_i X_i^L. See Note 1 for explanation of function logic.

    :param X:  L x n matrix, where L is the number of samples and n is the number of features
    :param w: n x 1 vector
    :return: L x 1 vector
    """
    return np.matmul(X, w)


# tested
def get_y_predictions(X, w):
    """
    Obtains predicted labels for all L samples, using the following formula:

    P(Y^L=1|x,w) = exp(A) / 1 + exp(A)
    where
    - i is the ith feature
    - L is the number of samples
    - A = w_0 + SUM_i (w_i X_i^L)

    :param X: L x n matrix, where L is the number of samples and n is the number of features
    :param w: n x 1 vector
    :return:  L x 1 vector
    """
    num_rows = X.shape[0]

    Xw = _calc_inner(X, w)
    top = np.exp(Xw)
    ones = np.ones(num_rows)
    bottom = ones + top
    return top / bottom


# tested
def _calc_gradient(X, y_true, y_pred):
    """
    Calculates the gradient. See Note 2 for explanation of function logic.

    :param X: L x n matrix, where L is the number of samples and n is the number of features
    :param y_true: L x 1 vector
    :param y_pred: L x 1 vector
    :return: Gradient in the form of an n x 1 vector
    """
    y_err = y_true - y_pred
    return np.matmul(X.T, y_err)


# tested
def _calc_left_half_log_likelihood(X, y_true, w):
    """
    Calculates the YA sum used in log likelihood, where A = w_0 + SUM_i^n w_i X_i^L.
    See Note 3 for details.

    :param X: L x n matrix, where L is the number of samples and n is the number of features
    :param y_true: L x 1 vector
    :param w: n x 1 vector
    :return: scalar
    """
    Xw = _calc_inner(X, w)
    return np.dot(y_true, Xw)


# tested
def _calc_right_half_log_likelihood(X, w):
    """
    Calculates the ln(1 + exp(A)) sum used in log likelihood, where A = w_0 + SUM_i^n w_i X_i^L.
    See Note 3 for details.

    :param X: L x n matrix, where L is the number of samples and n is the number of features
    :param w: n x 1 vector
    :return: scalar
    """
    Xw = _calc_inner(X, w)
    num_rows = X.shape[0]
    ones = np.ones(num_rows)  # for each sample
    inner = ones + np.exp(Xw)
    ln_Xw = np.log(inner)
    return np.sum(ln_Xw)  # sum over L samples


# tested
def _calc_log_likelihood(X, y_true, w):
    """
    Calculates log likelihood. See Note 3 for explanation of function logic.

    :param X: L x n matrix, where L is the number of samples and n is the number of features
    :param y_true: L x 1 vector
    :param w: n x 1 vector
    :return: scalar
    """
    # left half of expression
    sum_1 = _calc_left_half_log_likelihood(X, y_true, w)

    # right half of expression
    sum_2 = _calc_right_half_log_likelihood(X, w)

    return sum_1 - sum_2


def gradient_descent(X, y_true, w, eta, epsilon):
    """
    Performs gradient descent to derive optimal regression coefficients.

    :param X: L x n matrix, where L is the number of samples and n is the number of features
    :param y_true: L x 1 vector
    :param w: n x 1 vector
    :param eta: learning rate
    :param epsilon: convergence threshold
    :return: n x 1 vector
    """
    # set initial weights
    weights = w

    # calculate original log likelihood
    prev_log_likelihood = _calc_log_likelihood(X, y_true, weights)

    # perform gradient descent
    count = 0
    diff = np.Inf
    while diff > epsilon:

        count += 1
        if count > 100000:
            break  # stop descending because something is probably wrong

        # update weights
        y_pred = get_y_predictions(X, weights)
        gradient = _calc_gradient(X, y_true, y_pred)
        weights = _update_weights(weights, eta, gradient)

        # calculate difference
        log_likelihood = _calc_log_likelihood(X, y_true, weights)
        diff = np.abs(prev_log_likelihood - log_likelihood)

        # save log likelihood for next round
        prev_log_likelihood = log_likelihood

    print('count of rounds', count)
    return weights
