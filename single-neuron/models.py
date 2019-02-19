# Author: Dimitrios Damopoulos
# MIT license (see LICENCE.txt in the top-level folder)

import numpy as np
import numpy.random as random
import math_utils as M

class Neuron(object):
    """ A trainable model of a single neuron """

    def __init__(self, dims, param_initializer=random.randn):
        """
        Args:
            dims (int): dimensionality of the feature space
            param_initializer (function): a function that accepts an int as a 
                parameter and it returns a numpy vector of size equal to 
                ``dims''. This function is used for initializing the values of 
                both the weights of the neuron and its bias. For example, you 
                may supply here numpy.zeros or numpy.random.rand.
        """
        # We group all the parameters in one vector
        self.theta = param_initializer(dims + 1)

    @classmethod
    def description(cls):
        """ Returns a human-readable description of the class """

        raise NotImplementedError

    @classmethod
    def loss(cls, y_hat, y):
        """ 
        The loss function
        
        Args:
            y_hat (np.ndarray of shape N,): The vector of the predictions
            y (np.ndarray of shape N,): The vector of the ground truth values

        Returns:
            The value of the loss function
        """
        raise NotImplementedError

    def bias(self):
        return self.theta[0]

    def weights(self):
        return self.theta[1:]

    def activation(self, stimu):
        """
        The activation function

        Args:
            stimu (np.ndarray of shape N,): The stimulation of the neuron, i.e. 
                the weighted average of its inputs
        Returns:
            The value of the activation function
        """
        raise NotImplementedError

    def gradient_of_loss(self, Xn, y):
        """
        The gradient of the loss function with respect to the parameters of the 
        model
     
        Args:
            Xn (np.ndarray of shape N,m): The values of the features for the 
                sample points that the gradient should be calculated. 
                **Important note**: The first column of Xn should have only 1s.
            y (np.ndarray of shape N,): The ground truth values of the target 
                variable for the given samples.
        Returns:
            The sum of the gradients of the loss function for all the supplied 
            samples
        """
        raise NotImplementedError

    def predict(self, X, y=None):
        """
        Makes a prediction for the target value given the current values of the 
        parameters of the model

        Args:
            Xn (np.ndarray of shape N,m): The values of the features for the 
                sample points that the gradient should be calculated. 
            y (optional, np.ndarray of shape N,): The ground truth values of the 
                target variable for the given samples. 
        Returns:
            An nd.array of shape N, with the predictions of the model. If the 
            'y' has been provided, then it additionally returns the sum of the 
            loss for the given samples.
        """
        stimu = X.dot(self.weights()) + self.bias()
        y_hat = self.activation(stimu)

        if y is None: 
            return y_hat 
        else: 
            return y_hat, self.loss(y_hat, y) 

    def fit(self, X, y, lrate, epochs, on_epoch_end_callback=None):
        """ Training the neuron using gradient descent on the loss function
        
        Args:
            X (np.ndarray of shape (N, self.dims)): training set, features
            y (np.ndarray of shape (N,): training set, target values
            lrate (float): learning rate
            epochs (int): number of gradient descent iterations
            on_epoch_end_callback (function): A function that accepts as an 
                argument an integer (the index of current iterator) and a 
                LinearActivationNeuron object. This function can be used as a 
                callback in order to perform some action at the end of the 
                training iteration.

        Note:
            Currently, for all the implemented activations, the loss function 
            is convex with respect to all the weights and the bias. That means 
            that the only reason to use mini-batch gradient descent here would 
            be the size of the dataset. Given however that this is currently 
            passed through a CSV file, it is unlikely to be large enough to 
            cause concerns with the memory footprint. Hense, we perform 
            standard gradient descent here (sometimes called ``batch'' 
            gradient descent).
        """
        Xn = np.insert(X, 0, 1, axis=1)
        for e in range(epochs):
    
            self.theta -= lrate * self.gradient_of_loss(Xn, y)
            
            if on_epoch_end_callback:
                on_epoch_end_callback(e, self)

        if not np.isfinite(self.theta).all():
            print('\nAt least one parameter has an invalid value after fitting '
                  'the neuron model. That might be caused by a too large '
                  'learning rate and/or by having the \'normalize\' option disabled.\n')


class LinearNeuron(Neuron):
    """ A neuron with an identity activation function, trained on RMSE """

    def activation(self, stimu):
        return stimu

    @classmethod
    def description(cls):
        return 'Neuron with an identity activation function (trained on RMSE)'

    @classmethod
    def loss(cls, y_hat, y):
        return M.rmse(y_hat, y)

    def gradient_of_loss(self, Xn, y):

        stimu = Xn.dot(self.theta)
        return M.gradient_of_rmse(stimu, y, Xn)


class ReluNeuron(Neuron):
    """ A neuron with a ReLU activation function, trained on the RMSE error """

    def activation(self, stimu):
        return np.maximum(np.zeros(stimu.shape), stimu)

    @classmethod
    def description(cls):
        return 'Neuron with a ReLU activation function'

    @classmethod
    def loss(cls, y_hat, y):
        return M.rmse(y_hat, y)

    def gradient_of_loss(self, Xn, y):

        stimu = Xn.dot(self.theta)

        # The difference with the linear activation is that we take into account 
        # only the positive stimuli in order to compute the gradient.
        mask = stimu > 0
        stimu_masked = stimu[mask]
        y_masked = y[mask]
        Xn_masked = Xn[mask]

        N = stimu.shape[0]
        N_masked = stimu_masked.shape[0]
        
        if N_masked > 0: 
            gradient_masked = M.gradient_of_rmse(
                                            stimu_masked, y_masked, Xn_masked) 

            # we have to rescale, since so far the gradient has been calculated as 
            # the mean over only the positive stimuli.
            gradient_all =  gradient_masked * N_masked / N
        else:
            gradient_all = 0

        return gradient_all


class SigmoidNeuron(Neuron):
    """ 
    A model of a neuron with a sigmoid activation function trained on the 
    log-likelihood loss (which is basically the cross-entropy).
    """

    def activation(self, stimu):
        return M.sigmoid(stimu)

    @classmethod
    def description(cls):
        return 'Neuron with a sigmoid activation function'

    @classmethod
    def loss(self, y_hat, y):
        N = y.shape[0]
        assert N > 0, ('At least one sample is required in order to compute the '
                      'log-likehood loss')
        return M.cross_entropy(y_hat, y) / N 

    def gradient_of_loss(self, Xn, y):

        N = y.shape[0]
        assert N > 0, ('At least one sample is required in order to compute the '
                      'gradient of the log-likehood loss')

        stimu = np.dot(Xn, self.theta)
        y_hat = self.activation(stimu) 

        losses = y_hat - y 
        gradient = Xn.T.dot(losses) / N

        return gradient

def predict_with_neuron(model_class, X_train, y_train, X_valid, 
                              lrate, epochs, on_epoch_end_callback=None):
    """
    Args:
        X_train (np.ndarray of shape (N, m): Features of the training set
        y_train (np.ndarray of shape (N,): Target values of the training set
        X_valid (np.ndarray of shape (V, m): Features of the validation set
        lrate (float): learning rate
        epochs (int): number of epochs
        on_epoch_end_callback (function): A function that accepts as an 
            argument an integer (the index of current iterator) and a 
            LinearActivationNeuron object. This function can be used as a 
            callback in order to perform some action at the end of every epoch 

    Returns:
        The predictions of the trained neuron for X_valid and a np.ndarray 
        vector with the parameters of the trained neuron.
    """
    dims = X_train.shape[1]
    model = model_class(dims) 
    model.fit(X_train, y_train, lrate, epochs, on_epoch_end_callback)
    predictions = model.predict(X_valid)
    parameters = model.theta
    
    return predictions, parameters

def predict_with_linear_regression(X_train, y_train, X_valid):
    """
    Args:
        X_train (np.ndarray of shape (N, m): Features of the training set
        y_train (np.ndarray of shape (N,): Target values of the training set
        X_valid (np.ndarray of shape (V, m): Features of the validation set

    Returns:
        The predictions of linear regression for X_valid and a np.ndarray 
        vector with the parameters of the trained model.
    """
    from sklearn.linear_model import LinearRegression as LR

    model = LR()
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    parameters = np.insert(model.coef_, 0, model.intercept_)
    
    return predictions, parameters
