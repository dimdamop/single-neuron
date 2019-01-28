# Synopsis
`neuron-regress.py`: A python script for assessing the performance of a single linear artificial neuron model when used for a regression task and for comparing it with that of linear regression. 

# Syntax
```
neuron-regress.py [-h] --csv FILENAME [--target-name NAME]
                  [--validation-set-ratio RATIO] [--epochs EPOCHS]
                  [--lrate LRATE] [--no-norm] [--loglog]
```

# Description

The neuron model gets trained with batch gradient descent on the root mean square (RMSE) loss. The performance achieved with the scikit-learn implementation of linear regression is used for comparison. In particular, `neuron-regress.py` compuses the RMSE on a validation set at the end of every gradient descent iteration and it finally generates a plot with the progression of these measurements. If the training of the neuron goes fine, these measurements should converge towards the performance of linear regression on the same validation set. The latter is displayed as a dotted horizontal line on the same plot.

## Command line arguments

There is just one required argument (the --csv), the rest are optional. In particular:

* -h, --help
Show a detailed help message and exit.
* --csv FILENAME
**Required**. Relative path to the CSV file with the dataset
* --target-name NAME
The name of the variable to be regressed. If none is given then the last variable of the CSV is used as such.
* --validation-set-ratio RATIO
The fraction of the dataset to be used for validation (default: 0.1).
* --epochs EPOCHS
Number of epochs (iterations) to use for gradient descent (default: 200).
* --lrate LRATE
Learning rate to use for gradient descent (default: 0.1)
* --no-norm
By default, the features are linearly rescaled to so that the training set has a zero mean and a standard deviation of one. This switch prevents this default behavior (default: False).
*  --loglog
Use a logarithmic scale for the two axes of the plot. This can be useful if you have specified many epochs (default: False).

