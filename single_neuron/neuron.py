#!/usr/bin/env python

# Author: Dimitrios Damopoulos
# MIT license (see LICENCE.txt in the top-level folder)

"""
Batch gradient descent on a model of a single artificial neuron and plotting of 
the progression of the loss on some validation set during the gradient descent 
iterations.

This is a standalone script, i.e. it can be invoked directly in the command
line. Consult the accompanying README.md for details.
"""

from sys import stdout as cout
import time
from math import floor
import numpy as np
import numpy.random as rand
import single_neuron.models as models
import single_neuron.math_utils as M


class LossRecorder(object):
    """ Records the loss of some method for different training iterations """

    def __init__(self, X, y, epochs): 
        """
        Args:
            X (np.ndarray of shape N, m): The features of the set that the 
                loss will be calculated against.
            y (np.ndarray of shape N,): The target values of the set that the 
                loss will be calculated against.
        """
        self.X = X 
        self.y = y
        self.losses = np.full(epochs, np.nan)

    def record(self, epoch, model):
        """ 
        Records a measurement
        
        Args:
            epoch (int): the index of the training iteration this measurement 
                corresponds to
            model: An object that has a member function named ``prediction'' 
                that can accept the self.X np.ndarray as an argument (for 
                example, a Neuron object).
        """
        _, losses = model.predict(self.X, self.y)
        self.losses[epoch] = losses

def parse_csv(fn, sep=',', name_delim=None):
    """
    Parses a CSV file and returns the extracted values as an np.ndarray and a 
    list with the names of the columns

    Args:
        fn (str): path to the CSV file
        sep (char): Character that separates values in CSV within the same 
            line. It cannot be one of the following characters: '.' of '-'. 
            Default value: ','
        name_delim(str): A character that encapsulates the names of the columns
            in the header line of the  CSV file and which should be removed in 
            order to get the names of the columns. This parameter is useful if 
            the names of the columns are provided in a header line in a manner 
            like "col1", "col2", "col3" (in which case the ``name_delim'' should 
            be '"'. If ``name_delim'' is None, then no character is removed from 
            the names. It cannot be one of the following characters: '.' of '-'. 
            Default value: None
    
    Notes:
        Leading or trailing whitespace characters are removed from the column 
        names.
        The `sep' or the `name_delim' cannot appear in the values of the CSV.


    Returns:
        A np.ndarray with the values found in fn and a list with the names of 
        the columns 
    """

    assert sep != '.' and sep != '-', \
                        "The separating character cannot be a '.' or a '-'."
    lines = []

    with open(fn) as fd:
        for line in fd:

            # remove whitespace character from the beginning and the end
            line = line.strip()

            # ignore comments and empty lines
            if len(line) == 0 or line[0] == '#':
                continue

            lines.append(line)

    assert len(lines) > 1, 'No lines found in {0}'.format(fn)

    # The first line has the names of the columns
    names = lines.pop(0).split(sep)

    if name_delim is not None:
        names = [ name.strip(name_delim).strip() for name in names ]

    N = len(lines)
    m = len(names)
    values = np.zeros((N, m))

    for i in range(N):
        line = lines[i]
        entries = line.split(sep)
        values[i] = [ float(entry) for entry in entries ]

    return values, names

def train_valid_split(dataset, target_idx, validation_set_ratio, norm_features, 
                      modify_dataset=False):
    """
    Args:
        dataset (np.ndarray of shape N,m): The whole dataset where the training 
            and the validation set should be taken from.
        target_idx (int): The index of the target variable in the dataset
        validation_set_ratio (float): A number between 0 and 1 indicating the 
            percentage of the dataset that should be employed as the validation 
            set for the generation of the plot
        modify_dataset (bool): A flag that specifies whether the implementation 
            is allowed to change the ``dataset'' argument. If it is not allowed 
            to do so, then it might have to internally copy the dataset, which 
            might increase the memory requirements considerably, depending on 
            the size of the dataset. Default: False.
    
    Returns:
        A list with the following four elements: 

        X_train (np.ndarray of shape N1,m): The features of the training set
        y_train (np.ndarray of shape N1,): The target values of the training set
        X_valid (np.ndarray of shape N2,m): The features of the validation set
        y_valid (np.ndarray of shape N2,(): The target values of the validation 
            set.
    """

    if modify_dataset:
        values = dataset
    else:
        values = np.copy(dataset)

    # shuffle the dataset (for the sampling that follows) and separate it into 
    # features and a target variable
    rand.shuffle(values)
    N, m = values.shape
    feature_idxs = list(range(m))
    del feature_idxs[target_idx]
    X = values[:, feature_idxs]
    y = values[:, target_idx]

    # if it is a classification task, we 
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

    return X_train, y_train, X_valid, y_valid

def plot(method_names, losses_method_a, loss_method_b=None, 
         title='', xlabel='', ylabel='', loglog=False):
    """ 
    Displays a plot of the progression of the validation loss during training.
   
    Args:
        method_names (an iteratable object with at least one string): 
            method_names[0] is the name of the method that corresponds to 
            ``losses_method_a'' and method_names[1] (if supplied) is the name 
            of the method that corresponds to ``loss_method_b''
        losses_method_a (an np.ndarray vector): The values of the plot of the 
            first method
        loss_method_b (float): The value of the second method, which will 
        correspond to the height of the plotted horizontal line. If the length 
        of ``method_names'' is 1, this argument is ignored.
        title (str): The title of the figure. Default value: ''
        xlabel (str): The title of the horizontal axis. Default value: ''
        ylabel(str): The title of the vertical axis. Default value: ''
        loglog(bool): Whether to use a logarithmic scale for the two axes. 
            Default value: False
    """
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

    if len(method_names) > 1:
        # loss with the second method
        ax.plot((0, len(losses_method_a)), (loss_method_b, loss_method_b), 
                '--r', label=method_names[1])

    ax.legend(fontsize=15)
    plt.show()

def main(csv_fn, target_name, neuron_class, validation_set_ratio, lrate, epochs, 
         norm_features=True, lr_baseline=False, loglog=False):
    """
    Args: 
        csv_fn (str): The path to the CSV file with the dataset
        target_name (str): The name of the variable whose value is to be 
            predicted. If ``None'', the variable corresponding to the last 
            column of ``csv_fn'' is treated as the target variable.
        validation_set_ratio (float): A number between 0 and 1 indicating the 
            percentage of the dataset that should be employed as the validation 
            set for the generation of the plot.
        lrate (float): Learning rate for the gradient descent
        epochs (int): Number of gradient descent iterations
        norm_features (bool): Whether the values of the feature set should 
            be linearly rescaled so that the training set has zero mean and 
            a standard deviation of one. Default value: True.
    """
    
    dataset, names = parse_csv(csv_fn)

    # If a name for the target variable has been provided, we locate its index, 
    # otherwise we treat the last variable in the csv_fn as the target one.
    if target_name:
        try: 
            target_idx = names.index(target_name)
        except ValueError:
            print('No feature named \'{0}\' was found in the '
                  'dataset stored in {1}.'.format(target_name, csv_fn))
            print('The found names are the: {0}'.format(names))
            return
    else:
        target_idx = len(names) - 1
        target_name = names[-1]

    # sample the training and validation sets
    X_train, y_train, X_valid, y_valid = train_valid_split(dataset, target_idx, 
                                        validation_set_ratio, norm_features, 
                                        modify_dataset=True)

    # an object to record the loss on the validing set after every epoch
    recorder = LossRecorder(X=X_valid, y=y_valid, epochs=epochs)
    
    print('\nSize of training set: {0}'.format(X_train.shape[0]))
    print('Size of validation set: {0}'.format(X_valid.shape[0]))
    print('Dimensionality of feature space: {0}\n'.format(X_train.shape[1]))
    # predict with the neuron model
    cout.write('Training... ')
    cout.flush()
    cpu_start = time.clock()
    neuron_y_hat, neuron_params = models.predict_with_neuron(neuron_class, 
                                    X_train, y_train, X_valid, 
                                    lrate, epochs, recorder.record)
    cpu_end = time.clock()
    cout.write('ok. CPU elapsed time: {0:.3f} s\n\n'.format(
                                                        cpu_end - cpu_start))
    
    neuron_descr = neuron_class.description()
    neuron_final_loss = neuron_class.loss(neuron_y_hat, y_valid)
    print('The loss with {0} is: {1}'.format(neuron_descr, neuron_final_loss))
    print('The parameteres of {0} are: \n{1!s}'.format(neuron_descr, 
                                                               neuron_params))

    if lr_baseline:
        lr_y_hat, lr_params = models.predict_with_linear_regression(
                                                    X_train, y_train, X_valid)
        lr_loss = M.rmse(neuron_y_hat, y_valid)
        lr_name = 'Linear Regression model'
        print('The RMSE loss with a {0} is: {1}'.format(lr_name, lr_loss))
        print('The parameteres of the {0} are: \n{1!s}\n'.format(
                                                          lr_name, lr_params))
        method_names = (neuron_descr, lr_name)
    else:
        method_names = (neuron_descr, )
        lr_loss = None

    xlabel = 'epoch' 
    ylabel = 'Loss (on the validation set)' 
    title = 'Loss when a {0} is used to predict "{1}" (learning rate {2})'.format(
                                            neuron_descr, target_name, lrate)

    # show in a plot the loss of a single neuron model during gradient descent.
    # Optionally, also plot a horizontal line with the performance achieved 
    # with linear regression.
    plot(method_names, recorder.losses, lr_loss, 
         title=title, xlabel=xlabel, ylabel=ylabel, loglog=loglog)


if __name__ == '__main__':
    """ Entry point, when the script is invoked for standalone execution """

    import argparse

    activation_options = { 'identity' : models.LinearNeuron, 
                           'relu' : models.ReluNeuron, 
                           'sigm' : models.SigmoidNeuron }

    np.set_printoptions(precision=4, suppress=True)

    parser = argparse.ArgumentParser(
                description='Supervised learning using a single neuron model, '
                            'trained with batch gradient descent.', 
                formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
                epilog='See the README.md file for more details')

    parser.add_argument('--csv', 
                        help='Relative path to the CSV file with the dataset', 
                        type=str, 
                        metavar='FILENAME', 
                        dest='csv_fn', 
                        required=True)
    
    parser.add_argument('--target-name', 
                        help='The name of the target variable. If it is not '
                        'specified then the last column of the supplied CSV '
                        'file is used as such', 
                        default=None, 
                        type=str, 
                        metavar='NAME')
    
    parser.add_argument('--validation-set-ratio', 
                        help='The fraction of the dataset to be used for '
                              'validation', 
                        default=0.1, 
                        type=float, 
                        metavar='RATIO')

    parser.add_argument('--activation', 
                        help='Which activation function to use for the neuron '
                             'model. For the identity activation function and '
                             'for ReLU, the loss function that will be used is'
                             ' the RMSE (Root Mean Square Error). For the '
                             'sigmoid, it will the logarithm of the likelihood '
                             'function.', 
                        choices = activation_options, 
                        default = 'identity')

    parser.add_argument('--epochs', 
                        help='Number of epochs (iterations) to use for '
                             'gradient descent', 
                        default=200, 
                        type=int)

    parser.add_argument('--lrate', 
                        help='Learning rate to use for gradient descent', 
                        default=0.1, 
                        type=float)

    parser.add_argument('--no-norm', 
                        help='By default, each of the features is linearly '
                             'rescaled to so that its value in the training set'
                             ' has a zero mean and a standard deviation of one.'
                             ' This switch prevents this default behavior',  
                        action='store_true', 
                        default=False)

    parser.add_argument('--lr-baseline', 
                        help='Optionally, you can specify that you want the ' 
                             'performance of the trained model to be compared '
                             'with the one achieved with linear regression. '
                             'This will be shown as a horizontal line in the '
                             'generated plot.', 
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
         neuron_class=activation_options[args.activation], 
         lrate=args.lrate, 
         epochs=args.epochs, 
         norm_features=not(args.no_norm), 
         lr_baseline=args.lr_baseline, 
         loglog=args.loglog)
