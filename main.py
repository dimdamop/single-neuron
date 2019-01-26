import numpy as np
import numpy.random as rand

class LinearActivationNeuron(object):

    def __init__(self, dims, param_initializer=rand.randn):
        self.weights = param_initializer(dims)
        self.bias = param_initializer(1)[0]

    def response(self, X):
        y_hat = X.dot(self.weights) + self.bias
        return y_hat

    def RMSE_gradient_descent(self, X, y, lrate, epochs):

        N = X.shape[0]
        # include the intercept, so that we don' t have to add the bias each time
        Xn = np.insert(X, 0, 1, axis=1)
        theta = np.insert(self.weights, 0, self.bias, axis=0)

        for i in range(epochs):

            y_hat = Xn.dot(theta)
            losses = y - y_hat
            # the gradient of the root mean square error loss for our model
            gradient_of_loss = - 2 * Xn.T.dot(losses) / N
            theta -= lrate * gradient_of_loss

        self.bias = theta[0]
        self.weights = theta[1:]


def predict_with_linear_neuron(X_train, y_train, X_test, lrate, epochs):

    dims = X_train.shape[1]
    neuron = LinearActivationNeuron(dims) 
    neuron = LinearActivationNeuron(dims, param_initializer=np.zeros) 
    neuron.RMSE_gradient_descent(X_train, y_train, lrate, epochs)
    return neuron.response(X_test)


def predict_with_linear_regression(X_train, y_train, X_test):

    from sklearn.linear_model import LinearRegression as LR

    lr_model = LR().fit(X_train, y_train)
    return lr_model.predict(X_test)


def parse_csv(fn, sep=',', name_delim='"'):

    lines = open(fn).readlines()
    assert len(lines) > 0, 'No lines found in {0}'.format(fn)

    # The first line has the names of the columns
    names = lines.pop(0).split(sep)
    names = [ name.strip(name_delim) for name in names ]

    N = len(lines)
    m = len(names)
    values = np.zeros((N, m))

    for i in range(N):
        line = lines[i]
        entries = line.split(sep)
        values[i] = [ float(entry) for entry in entries ]

    return values, names


def main(csv_fn, target_name, testing_set_ratio, lrate, epochs, normalize_features=True):
    """
        normalize_features: linearly rescale 
    """

    values, names = parse_csv(csv_fn)

    # If a name for the target variable has been provided, we find its index,  
    # otherwise we treat the last variable in the csv_fn as the target one.
    if target_name:
    	target_idx = values.index(target_name)
    else:
	target_idx = len(names) - 1
    
    if target_idx < 0:
        print('No feature named \'{0}\' was found in the dataset {1}.'.format(target_name, csv_fn))
        print('The found names are the: {0}'.format(names))
        return

    N, m = values.shape
    feature_idxs = list(range(m))
    del feature_idxs[target_idx]

    # shuffle the dataset (for the sampling that follows) and separate it into 
    # features and a target variable.
    rand.shuffle(values)
    X = values[:, target_idx]
    y = values[:, features_idxs]

    # sample training and testing sets TODO: round is not ok
    test_n = int(round(N * testing_set_ratio))
    X_test = X[:test_n]
    y_test = y[:test_n]
    X_train = X[test_n:]
    y_train = y[test_n:]

    if normalize_features:

        # we normalize based on the statistics of the training set only
        mu = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)

        X_train = (X_train - mu) / std
        X_test = (X_test - mu) / std

    # predict with the two methods
    y_pred_ne = predict_with_linear_neuron(X_train, y_train, X_test, lrate, epochs)
    y_pred_lr = predict_with_linear_regression(X_train, y_train, X_test)

    print('NE: {0}'.format(np.abs(y_pred_ne - y_test).mean()))
    print('LR: {0}'.format(np.abs(y_pred_lr - y_test).mean()))

if __name__ != None:
    
    main(csv_fn='mass_boston.csv', 
         testing_set_ratio=0.1, 
         target_name='medv',
         lrate=0.1, 
         epochs=10000)
