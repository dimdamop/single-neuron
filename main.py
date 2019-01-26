from math import floor
import numpy as np
import numpy.random as rand
from numpy.linalg import norm


class LinearActivationNeuron(object):

    def __init__(self, dims, param_initializer=rand.randn):
        # We group all the parameters in one vector
        self.theta = param_initializer(dims + 1)

    def bias(self):
        return self.theta[0]

    def weights(self):
        return self.theta[1:]

    def predict(self, X):
        y_hat = X.dot(self.weights()) + self.bias()
        return y_hat

    def fit(self, X, y, lrate, epochs, on_epoch_end_callback=None):

        N = X.shape[0]
        assert N > 0, 'At least one training example is required'

        # we add an intercept, to avoid the separate treatment of the bias
        Xn = np.insert(X, 0, 1, axis=1)

        for e in range(epochs):

            y_hat = Xn.dot(self.theta)
            losses = y - y_hat
            # the gradient of the RMSE
            gradient_of_loss = - 2 * Xn.T.dot(losses) / N
            self.theta -= lrate * gradient_of_loss
            
            if on_epoch_end_callback:
                on_epoch_end_callback(e, self)

        if not np.isfinite(self.theta).all():
            print('\nAt least one parameter has an invalid value after fitting '
                  'the neuron model. That might be caused by a too large '
                  'learning rate and/or by having the \'normalize\' option disabled.\n')


def predict_with_linear_neuron(X_train, y_train, X_test, lrate, epochs, 
                        on_epoch_end_callback=None):

    dims = X_train.shape[1]
    model = LinearActivationNeuron(dims, param_initializer=np.zeros) 
    model.fit(X_train, y_train, lrate, epochs, on_epoch_end_callback)
    predictions = model.predict(X_test)
    parameters = model.theta
    
    return predictions, parameters


def predict_with_linear_regression(X_train, y_train, X_test):

    from sklearn.linear_model import LinearRegression as LR

    model = LR()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    parameters = np.insert(model.coef_, 0, model.intercept_)
    
    return predictions, parameters

def parse_csv(fn, sep=',', name_delim='"'):

    lines = [ line.strip() for line in open(fn) ] 
    assert len(lines) > 1, 'No lines found in {0}'.format(fn)

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


class RmseRecorder(object):

    def __init__(self, X, y, epochs): 
        self.X = X 
        self.y = y
        self.losses = np.full(epochs, np.nan)

    def rmse(a, b):
        e = a - b
        mse = norm(e).mean()
        return np.sqrt(mse)

    def record(self, epoch, model):
        y_hat = model.predict(self.X)
        self.losses[epoch] = RmseRecorder.rmse(self.y, y_hat)


def plot(method_names, losses_method_a, loss_method_b, loglog=False, fn_out=None):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    # losses with the first method
    if loglog:
        ax.loglog(losses_method_a, '--b.', label=method_names[0])
    else:
        ax.plot(losses_method_a, '--b.', label=method_names[0])

    # loss with the second method
    ax.plot((0, len(losses_method_a)), (loss_method_b, loss_method_b) , 'r', label=method_names[1])

    ax.legend()
    plt.show()

    if fn_out:
        fig.savefig(fn_out)

def main(csv_fn, target_name, testing_set_ratio, lrate, epochs, normalize_features=True):
    """
        normalize_features: linearly rescale 
    """

    values, names = parse_csv(csv_fn)

    # If a name for the target variable has been provided, we locate its index, 
    # otherwise we treat the last variable in the csv_fn as the target one.
    if target_name:
        try: target_idx = names.index(target_name)
        except ValueError:
            print('No feature named \'{0}\' was found in the '
                  'dataset {1}.'.format(target_name, csv_fn))
            print('The found names are the: {0}'.format(names))
            return
    else:
        target_idx = len(names) - 1

    # shuffle the dataset (for the sampling that follows) and separate it into 
    # features and a target variable
    rand.shuffle(values)
    N, m = values.shape
    feature_idxs = list(range(m))
    del feature_idxs[target_idx]
    X = values[:, feature_idxs]
    y = values[:, target_idx]

    # sample training and testing sets
    test_n = int(floor(N * testing_set_ratio + .5))
    X_test = X[:test_n]
    y_test = y[:test_n]
    X_train = X[test_n:]
    y_train = y[test_n:]

    if normalize_features:
        # we normalize based on the statistics of only the training set
        mu = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)

        X_train = (X_train - mu) / std
        X_test = (X_test - mu) / std

    # an object to record the loss on the testing set after every epoch
    recorder = RmseRecorder(X=X_test, y=y_test, epochs=epochs)

    # predict with the two methods
    ne_pred, ne_params = predict_with_linear_neuron(X_train, y_train, X_test, 
                            lrate, epochs, recorder.record)
    lr_pred, lr_params = predict_with_linear_regression(X_train, y_train, X_test)

    method_names = ('a single neuron', 'linear regression')
    method_losses = (RmseRecorder.rmse(ne_pred, y_test), 
                     RmseRecorder.rmse(lr_pred, y_test))
    method_params = (ne_params, lr_params)

    # print results
    for name, loss, params in zip(method_names, method_losses, method_params):
        print('The loss with {0} is: {1}'.format(name, loss))
        print('The parameteres of {0} are: \n{1!s}\n'.format(name, params))

    # show how the loss of a single neuron model during gradient descent 
    # compares to that of LR
    plot(method_names, recorder.losses, method_losses[1])

if __name__ != None:

    np.set_printoptions(precision=4)

    main(csv_fn='dataset/mass_boston.csv', 
         testing_set_ratio=0.1, 
         target_name=None,
         lrate=0.1, 
         epochs=200)
