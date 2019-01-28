# Synopsis
`neuron-regress.py': A python script for assessing the performance of a single linear artificial neuron model when used for a regression task and for comparing it with that of linear regression. 

# Syntax
```
neuron-regress.py [-h] --csv FILENAME [--target-name NAME]
                     [--validation-set-ratio RATIO] [--epochs EPOCHS]
                     [--lrate LRATE] [--no-norm] [--loglog]
```

# Description

The neuron model gets trained with batch gradient descent on the root mean square (RMSE) loss. The performance achieved with the ``scikit-learn'' implementation of linear regression is used for comparison. In particular, `neuron-regress.py' compuses the RMSE on a validation set at the end of every gradient descent iteration and it finally generates a plot with the progression of these measurements. If the training of the neuron goes fine, these measurements should converge towards the performance of linear regression on the same validation set. The latter is displayed as a dotted horizontal line on the same plot.
