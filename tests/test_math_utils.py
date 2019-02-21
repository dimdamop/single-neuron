# Author: Dimitrios Damopoulos
# MIT license (see LICENCE.txt in the top-level folder)

import unittest 
import numpy as np
from numpy import random
from single_neuron import math_utils as math_utils

max_vector_len_to_test = 100

class SigmoidTestCase(unittest.TestCase):
    """
    sigmoid should accept vectors of arbirtary length. Expected output 
    for 0 is 0.5, for infinity is 1 and for negative infinity is 0. The 
    output should always be [0, 1].
    """
    def test_output_for_0(self):
        for z_len in range(1, max_vector_len_to_test):
            z = np.zeros(z_len)
            sigm = math_utils.sigmoid(z)
            self.assertAlmostEqual((sigm - 0.5).sum(), 0)

    def test_output_for_inf(self):
        for z_len in range(1, max_vector_len_to_test):
            z = np.full(z_len, np.inf)
            sigm = math_utils.sigmoid(z)
            self.assertAlmostEqual((sigm - 1).sum(), 0)

    def test_output_for_neg_inf(self):
        for z_len in range(1, max_vector_len_to_test):
            z = np.full(z_len, -np.inf)
            sigm = math_utils.sigmoid(z)
            self.assertAlmostEqual(sigm.sum(), 0)

    def test_output_between_0_1(self):
        for z_len in range(1, max_vector_len_to_test):
            z = random.rand(z_len)
            sigm = math_utils.sigmoid(z)
            self.assertTrue((sigm >= 0).all())
            self.assertTrue((sigm <= 1).all())


class RmseTestCase(unittest.TestCase):
    """
    RMSE should accept vectors of arbirtary length. The RMSE of a vector 
    with a copy of itself should be zero. rmse(a, b) should be equal to 
    rmse(b, a). rmse(a, 0)^2 should be equal to |a|^2 / len(a) .
    """
    def test_is_output_with_itself_zero(self):
        for y_len in range(1, max_vector_len_to_test):
            y = (500 - y_len) * random.rand(y_len)
            y_hat = np.copy(y)
            self.assertAlmostEqual(math_utils.rmse(y_hat, y), 0)

    def test_are_ab_ba_equal(self):
        for y_len in range(1, max_vector_len_to_test):
            y = (50 - y_len) * random.rand(y_len)
            y_hat = (500 - y_len) * random.rand(y_len)
            self.assertAlmostEqual(math_utils.rmse(y_hat, y), 
                                   math_utils.rmse(y, y_hat))

    def test_output_with_zero(self):
        for y_len in range(1, max_vector_len_to_test):
            y = (5 - y_len) * random.rand(y_len)
            y_hat = np.zeros(y_len)
            self.assertAlmostEqual(math_utils.rmse(y_hat, y), 
                                   np.sqrt(y.T.dot(y) / y.shape[0]))

class CrossEntropyTestCase(unittest.TestCase):
    """
    the cross-entropy of two distributions should be positive. The
    cross-entropy of an one-hot vector with itself is 0.
    """
    def test_is_positive(self):
        for y_len in range(1, max_vector_len_to_test):
            # the following lines contruct two random vectors with elements 
            # between [0, 1] which add up to 1.
            y1 = np.zeros(y_len)
            y2 = np.zeros(y_len)
            indices = random.permutation(np.arange(y_len))

            for i in indices[:-1]:
                remaining = 1 - y1.sum() 
                y1[i] = random.rand() * remaining
                
                remaining = 1 - y2.sum() 
                y2[i] = random.rand() * remaining

            y1[indices[-1]] = 1 - y1.sum()
            y2[indices[-1]] = 1 - y2.sum()

            self.assertTrue(math_utils.cross_entropy(y1, y2) >= 0)
            self.assertTrue(math_utils.cross_entropy(y2, y1) >= 0)

    def test_is_one_hot_with_itself_0(self):
        for y_len in range(1, max_vector_len_to_test):
            # construct an one-hot vector
            y = np.zeros(y_len)
            y[random.randint(y_len)] = 1

            y_hat = np.copy(y)
            self.assertAlmostEqual(math_utils.cross_entropy(y_hat, y), 0)

