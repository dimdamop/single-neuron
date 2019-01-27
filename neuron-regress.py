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


def predict_with_linear_neuron(X_train, y_train, X_valid, lrate, epochs, 
                        on_epoch_end_callback=None):

    dims = X_train.shape[1]
    model = LinearActivationNeuron(dims, param_initializer=np.zeros) 
    model.fit(X_train, y_train, lrate, epochs, on_epoch_end_callback)
    predictions = model.predict(X_valid)
    parameters = model.theta
    
    return predictions, parameters


def predict_with_linear_regression(X_train, y_train, X_valid):

    from sklearn.linear_model import LinearRegression as LR

    model = LR()
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
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

    @staticmethod
    def rmse(a, b):
        e = a - b
        mse = norm(e).mean()
        return np.sqrt(mse)

    def record(self, epoch, model):
        y_hat = model.predict(self.X)
        self.losses[epoch] = RmseRecorder.rmse(self.y, y_hat)


def plot(method_names, losses_method_a, loss_method_b, 
         title='', xlabel='', ylabel='', loglog=False):

    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)

    # losses with the first method
    if loglog:
        ax.loglog(losses_method_a, '-b.', label=method_names[0])
    else:
        ax.plot(losses_method_a, '-b.', label=method_names[0])

    # loss with the second method
    ax.plot((0, len(losses_method_a)), (loss_method_b, loss_method_b), '--r', 
            label=method_names[1])

    ax.legend(fontsize=15)
    plt.show()


def main(csv_fn, target_name, validation_set_ratio, lrate, epochs, 
         norm_features=True, loglog=False):
    """
        norm_features: linearly rescale 
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
        target_name = names[-1]

    # shuffle the dataset (for the sampling that follows) and separate it into 
    # features and a target variable
    rand.shuffle(values)
    N, m = values.shape
    feature_idxs = list(range(m))
    del feature_idxs[target_idx]
    X = values[:, feature_idxs]
    y = values[:, target_idx]

    # sample training and validation sets
    valid_n = int(floor(N * validation_set_ratio + .5))
    X_valid = X[:valid_n]
    y_valid = y[:valid_n]
    X_train = X[valid_n:]
    y_train = y[valid_n:]

    if norm_features:
        # we normalize based on the statistics of only the training set
        mu = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)

        X_train = (X_train - mu) / std
        X_valid = (X_valid - mu) / std

    # an object to record the loss on the validing set after every epoch
    recorder = RmseRecorder(X=X_valid, y=y_valid, epochs=epochs)

    # predict with the two methods
    ne_pred, ne_params = predict_with_linear_neuron(X_train, y_train, X_valid, 
                            lrate, epochs, recorder.record)
    lr_pred, lr_params = predict_with_linear_regression(X_train, y_train, X_valid)

    method_names = ('a single linear neuron', 'linear regression baseline')
    method_losses = (RmseRecorder.rmse(ne_pred, y_valid), 
                     RmseRecorder.rmse(lr_pred, y_valid))
    method_params = (ne_params, lr_params)

    # print results
    for name, loss, params in zip(method_names, method_losses, method_params):
        print('The loss with {0} is: {1}'.format(name, loss))
        print('The parameteres of {0} are: \n{1!s}\n'.format(name, params))

    # show in a plot how the loss of a single neuron model during gradient 
    # descent compares to that of LR
    title_patt = ('RMSE when regressing "{0}" using a linear neuron model '
                  '\ntrained with gradient descent (learning rate {1})')
    title = title_patt.format(target_name, lrate)
    xlabel = 'epoch' 
    ylabel = 'RMSE (on the validation set)' 

    plot(method_names, recorder.losses, method_losses[1], 
         title=title, xlabel=xlabel, ylabel=ylabel, loglog=loglog)


if __name__ != None:

    import argparse

    np.set_printoptions(precision=4, suppress=True)

    parser = argparse.ArgumentParser(
                description='Regression using a single linear neuron model, '
                            'trained with gradient descent.', 
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--csv', 
                        help='Relative path to the CSV file with the dataset', 
                        action='store', 
                        type=str, 
                        metavar='FILENAME', 
                        dest='csv_fn', 
                        required=True)
    
    parser.add_argument('--target-name', 
                        help='The name of the variable to be regressed. If none'
                             ' is given then the last variable of the CSV is'
                             ' used as such', 
                        action='store', 
                        default=None, 
                        type=str, 
                        metavar='NAME')
    
    parser.add_argument('--validation-set-ratio', 
                        help='The fraction of the dataset to be used for '
                              'validation', 
                        action='store', 
                        default=0.1, 
                        type=float, 
                        metavar='RATIO')

    parser.add_argument('--epochs', 
                        help='Number of epochs (iterations) to use for '
                             'gradient descent', 
                        action='store', 
                        default=200, 
                        type=int)

    parser.add_argument('--lrate', 
                        help='Learning rate to use for gradient descent', 
                        action='store', 
                        default=0.1, 
                        type=float)

    parser.add_argument('--no-norm', 
                        help='By default, the features are linearly rescaled to'
                             ' a zero mean and a standard deviation of one.'
                             ' This switch prevents this default behavior', 
                        action='store_true', 
                        default=False)

    parser.add_argument('--loglog', 
                        help='use a logarithmic scale for the two axes '
                              'of the plot', 
                        action='store_true', 
                        default=False)

    args = parser.parse_args()

    main(csv_fn=args.csv_fn, 
         validation_set_ratio=args.validation_set_ratio, 
         target_name=args.target_name,
         lrate=args.lrate, 
         epochs=args.epochs, 
         norm_features=not(args.no_norm), 
         loglog=args.loglog)
