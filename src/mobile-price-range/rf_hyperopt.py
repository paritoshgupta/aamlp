import numpy as np
import pandas as pd
import config
from functools import partial
from numpy.ma import MaskedArray
import sklearn.utils.fixes
sklearn.utils.fixes.MaskedArray = MaskedArray

from sklearn import metrics
from sklearn import ensemble
from sklearn import model_selection
import hyperopt
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope


def optimize(params, x, y):
    """
    The main optimizatiion function
    This function takes all the arguments from the search space and training features and target.
    It then initialized the models by setting the chosen paramaters and runs crossvalidation and returns a 
    negative accuracy score
    :param params: dict of params from hyperopt
    :param x: training data
    :param y: labels/targets
    :return negative accuracy after 5 folds
    """

    # initialize model with current parameters
    model = ensemble.RandomForestClassifier(**params)

    # initialize stratified k fold
    kf = model_selection.StratifiedKFold(n_splits=5)

    # initialize accuracy list
    accuracies = []

    # loop over all folds
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        # fit model on train data
        model.fit(xtrain, ytrain)

        # create predictions
        preds = model.predict(xtest)

        # calculate and append accuracy
        accuracy = metrics.accuracy_score(ytest, preds)
        accuracies.append(accuracy)

    # return negative accuracy
    return -1 * np.mean(accuracies)

if __name__ == "__main__":

    # read training data
    df = pd.read_csv(config.TRAINING_FILE)
    # drop target
    X = df.drop('price_range', axis=1).values
    y = df.price_range.values

    # define a parameter space
    # use hyperopt
    param_space = {
        # quniform gives round(uniform(low, high)/q) * q
        # we want int values for depth and estimators
        "max_depth": scope.int(hp.quniform("max_depth", 1, 15, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 1500, 1)),
        # choice from a list of values
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        # uniform choices a value between two values
        "max_features": hp.uniform("max_features", 0, 1)
    }

    # partial function
    optimization_function = partial(optimize, x=X, y=y)

    # initialize Trials to keep logging information
    trials = Trials()

    # run hyperopt
    hopt = fmin(fn=optimization_function,
    space=param_space,
    algo=tpe.suggest,
    max_evals=15,
    trials=trials)

    print(hopt)