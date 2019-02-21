# Author: Dimitrios Damopoulos
# MIT license (see LICENCE.txt in the top-level folder)

# TODO: documentation

import numpy as np

def sigmoid(z):
    sigm = 1. / (1. + np.exp(-z))
    return sigm

def cross_entropy(y_hat, y):

    # get values which are very close to zero a bit higher, in order to avoid
    # NaNs in the computation of the logarithm later.
    eps = np.finfo(float).eps
    y_hat[y_hat < eps] = eps

    ce = - np.log(y_hat).T.dot(y)
    return ce

def cross_entropy_with_one_hot_vector(y_hat, y):

    eps = np.finfo(float).eps
    y1 = y > 0
    y0 = np.logical_not(y1)
    part1 = np.log(y_hat[y1] + eps).sum()
    part0 = np.log(1. - y_hat[y0] + eps).sum()
    
    return -(part0 + part1)

def rmse(y_hat, y):
    losses = y_hat - y
    se = losses.T.dot(losses)
    return np.sqrt(se / y.shape[0])

def gradient_of_rmse(y_hat, y, Xn):
        
    N = y.shape[0]
    assert N > 0, ('At least one sample is required in order to compute the '
                  'RMSE loss')
   
    losses = y - y_hat
    gradient = - 2 * Xn.T.dot(losses) / N

    return gradient
