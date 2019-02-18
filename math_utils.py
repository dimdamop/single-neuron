import numpy as np
from numpy.linalg import norm

def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm

def cross_entropy(y_hat, y):

    eps = np.finfo(float).eps

    label1_threshold = (y.max() + y.min()) / 2
    y1 = y > label1_threshold
    y0 = np.logical_not(y1)
    part1 = np.log(sigmoid(y_hat[y1]) + eps).sum()
    part0 = np.log(1. - sigmoid(y_hat[y0]) + eps).sum()
    
    return -(part0 + part1)

def rmse(y_hat, y):
    losses = y_hat - y
    mse = norm(losses).mean()
    return np.sqrt(mse)

def gradient_of_rmse(y_hat, y, Xn):
        
    N = y.shape[0]
    assert N > 0, ('At least one sample is required in order to compute the '
                  'RMSE loss')
   
    losses = y - y_hat
    gradient = - 2 * Xn.T.dot(losses) / N

    return gradient
