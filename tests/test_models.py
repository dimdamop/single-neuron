# Author: Dimitrios Damopoulos
# MIT license (see LICENCE.txt in the top-level folder)

import unittest 
import numpy as np
from numpy import random
from numpy import linalg as LA

from sklearn.linear_model import LinearRegression, LogisticRegression

from single_neuron import models as models
from single_neuron import math_utils as math_utils

datasets_n = 50
max_ds_n = 10000
max_features_n = 100
max_abs_value = 1000
min_epochs = 100
max_epochs = 10000
min_lr = 1e-9
max_lr = 1e-5

def generate_synthetic_datasets(N_max, m_max, gaussian=False):

    N_train = random.randint(3, N_max + 1)
    N_valid = random.randint(3, N_max + 1)
    m = random.randint(2, m_max + 1)

    if gaussian:
        # we are generating a synthetic dataset based on a multivariate Gaussian 
        # distribution. In order to generate the latter, we need a mean vector 
        # (easy) and a positive definite matrix for the covariances. This matrix 
        # is way more tricky to sample and I don' t know what is the best way. 
        # My current brute-force approach is the following: (a) I sample m 
        # vectors; (b) I take all the possible inner products (Gram matrix) as 
        # the covariance matrix and (c) if the covariance matrix is singular, I 
        # go back to step (b). 
        
        mu = 2 * (random.rand(m) - 0.5) * max_abs_value
        
        Cov = np.zeros([m, m])
        while LA.matrix_rank(Cov) != m:
            a = 2 * (random.rand(m) - 0.5) * max_abs_value
            X = a * random.rand(m, m)
            Cov = X.T.dot(X)
       
        train_ds = random.multivariate_normal(mu, Cov, N_train)
        valid_ds = random.multivariate_normal(mu, Cov, N_valid)

    else:
        # uniformly random datasets
        train_ds = 2 * (random.rand(N_train, m) - 0.5) * max_abs_value
        valid_ds = 2 * (random.rand(N_valid, m) - 0.5) * max_abs_value

    return train_ds, valid_ds

            
class TestLinearNeuron(unittest.TestCase):

    def setUp(self):
        """
        Prepare a few synthetic datasets for the tests. Two categories of 
        datasets: One random without any implied structure and one that arises 
        from a predefined distribution.
        """

        self.train_X = []
        self.valid_X = []
        
        self.train_y = []
        self.valid_y = []
        
        for ds_i in range(0, datasets_n):
            
            # make sure that there are some datasets with extremely small values 
            if ds_i < 10:
                N_max = 7
            else:
                N_max = max_ds_n

            if ds_i < 10:
                m_max = 2
            else:
                m_max = max_features_n
            
            #gaussian = random.rand() < 0.5
            gaussian = True

            train_ds, valid_ds = generate_synthetic_datasets(N_max, m_max, 
                                                                    gaussian)

            # we use the last column as the target variable
            self.train_X.append(train_ds[:, :-1])
            self.valid_X.append(valid_ds[:, :-1])
            
            self.train_y.append(train_ds[:, -1])
            self.valid_y.append(valid_ds[:, -1])

        self.lin_model = LinearRegression()

    def test_rmse_is_equal_with_sklearn(self):
        pass

    def test_params_are_equal_with_sklearn(self):
        pass

    def test_initialization_does_not_matter(self):
        pass


class TestReluNeuron(unittest.TestCase):

    def test_rmse_is_equal_with_sklearn(self):
        pass

    def test_initialization_with_negatives_leads_to_zero_gradients(self):
        pass

    def test_initialization_does_not_matter(self):
        pass


class TestLogisticNeuron(unittest.TestCase):

    def test_ce_is_equal_with_sklearn(self):
        pass

    def test_initialization_does_not_matter(self):
        pass
