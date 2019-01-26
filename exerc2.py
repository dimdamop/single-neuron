import sys 
import numpy as np
from numpy import linalg as LA

max_float_value = sys.float_info.max
min_z_value = -np.log(max_float_value / 2)

def sigmoid(z):
    """ Computes the sigmoid of z element-wise.
    z: A numpy array. 
    Returns: A numpy array of the same shape as z with the values of the sigmoid function, calculated on z.
    """
    #z[z < min_z_value] = min_z_value

    return 1. / (1 + np.exp(-z))

def one_minus_sigmoid(z):
    """ Computes the complemtary values of the sigmoid of z (ie, 1 - sigmoid(z)) element-wise.
    z: A numpy array. 
    Returns: A numpy array of the same shape as z with the values of the sigmoid function, calculated on z.
    """
    exps = np.exp(-z)
    return exps / ( 1 + exps)

def log_likelihood(theta, X, y):
    """ Compute the logarithm of the likelihood of the data (X, y), assuming logistic model for the probability with parameters theta. This is functionally equivalent to np.sum(y * z - np.log(1 + np.exp(z)) but faster.
    theta: A numpy vector of the parameters of the logistic model.
    X: A numpy array with the features of the examples.
    y: A numpy vector with the ground truth of the examples. The values in this vector must be either 1 or 0.
    Returns: The logarithm of the logistic likelihood.
    """
    z = np.dot(X, theta)

    positive_examples = y > 0
    negative_examples = np.logical_not(positive_examples)

    positive_examples_part = np.log(sigmoid(z[positive_examples]))
    negative_examples_part = np.log(one_minus_sigmoid(z[negative_examples]))

    return positive_examples_part + negative_examples_part

def logistic_regression_predictions(theta, X):
    z = np.dot(X, theta)
    return sigmoid(z)

def gradient_of_log_of_logistic_likelihood(theta, X, y):

    y_predict = logistic_regression_predictions(theta, X)
    err_diff = y - y_predict

    return np.dot(X.T, err_diff), err_diff

def gradient_descent(X, y, num_steps, alpha, gradient_cost_fn):

    # initialize the parameters of the model with zeros
    theta = np.zeros(X.shape[1])
    losses = []

    # gradient descent on all the examples at every step
    for i in range(num_steps):
        grad, err = gradient_cost_fn(theta, X, y)

        if (i + 1) % 200 == 0:
            print('Iteration ' + str(i + 1) + '/' + str(num_steps) + '. Norm of gradient is ' + str(LA.norm(grad)) + '. Mean prediction error is ' + str(err.mean()) + '.')
        theta += alpha * grad 

    return theta


if __name__ != None:

    import tempfile
    import urllib.request as ur
    import pandas as pd

    # fetch the dataset. It is in a CSV format. 
    sys.stdout.write('Fetching training and testing dataset... ')
    sys.stdout.flush()
    train_file = tempfile.NamedTemporaryFile()
    test_file = tempfile.NamedTemporaryFile()
    ur.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', train_file.name)
    ur.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', test_file.name)
    
    # load the dataset in Pandas dataframes
    CSV_COLUMNS = [ 'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_bracket']
    df_train = pd.read_csv(train_file.name, names=CSV_COLUMNS, skipinitialspace=True) 
    df_test = pd.read_csv(test_file.name, names=CSV_COLUMNS, skipinitialspace=True, skiprows=1)
    sys.stdout.write('ok\n')

    # Since the task is a binary classification problem, we'll construct a label column named 'label' whose value is 1 if the income is over 50K, and 0 otherwise.
    sys.stdout.write('Computing target labels, normalizing the dataset... ')
    sys.stdout.flush()
    label_col_name = 'income_bracket'
    train_labels = (df_train[label_col_name].apply(lambda x: '>50K' in x)).astype(int)
    test_labels = (df_test[label_col_name].apply(lambda x: '>50K' in x)).astype(int)
    # now let' s drop this from the data
    df_train = df_train.drop(label_col_name, axis=1)
    df_test = df_test.drop(label_col_name, axis=1)
    # most of the predictors are categorical. We will explicity label those as such.
    categorical_columns_names = [ 'workclass', 'fnlwgt', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'native_country']
    
    for dataset in (df_train, df_test):
        for col_name in categorical_columns_names:
            dataset[col_name] = pd.Categorical(dataset[col_name])
            dataset[col_name] = dataset[col_name].cat.codes

    # convert the Pandas dataframes in numpy arrays
    X_train = df_train.as_matrix().astype(float)
    y_train = train_labels.as_matrix().astype(float)
    X_test = df_test.as_matrix().astype(float)
    y_test = test_labels.as_matrix().astype(float)

    x_max_values = X_train.max()
    x_min_values = X_train.min()
    X_train = np.divide(X_train - X_train.min(axis=0), X_train.max(axis=0) - X_train.min(axis=0))
    X_test = np.divide(X_test - X_test.min(axis=0), X_test.max(axis=0) - X_test.min(axis=0))
    sys.stdout.write('ok\n')

    num_iterations = 4000
    sys.stdout.write('-- Gradient descent (' + str(num_iterations) + ') iterations)\n')
    sys.stdout.flush()
    # Use SGD in order to find the parameters of the logistic regression model
    theta = gradient_descent(X=X_train, 
                             y=y_train, 
                             num_steps=num_iterations, 
                             alpha=0.0001, 
                             gradient_cost_fn=gradient_of_log_of_logistic_likelihood)
    y_test = y_test > 0
    test_n = y_test.shape[0]
    test_pos = y_test.sum()
    test_neg = test_n - test_pos

    sys.stdout.write('-- Evaluating the trained classifier on the testing set samples)\n')
    sys.stdout.write('There are ' + str(test_n) + ' testing samples. ' + str(test_pos) + ' of them are positive and ' + str(test_neg) + ' are negative.\n')
    sys.stdout.flush()

    y_test_prob_predict = logistic_regression_predictions(theta, X_test)
    y_test_predict = y_test_prob_predict > 0.5

    tpr = np.logical_and(y_test_predict, y_test).sum() / float(test_pos)
    tnr = np.logical_and(np.logical_not(y_test_predict), np.logical_not(y_test)).sum() / float(test_neg)

    acc = (y_test_predict == y_test).sum() / float(test_n)

    print('True positive rate: ' + str(tpr))
    print('True negative rate: ' + str(tnr))
    print('Accuracy: ' + str(acc))
