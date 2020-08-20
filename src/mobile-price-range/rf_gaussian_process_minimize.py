import numpy as np
import pandas as pd
import config
from functools import partial
from numpy.ma import MaskedArray
import sklearn.utils.fixes
sklearn.utils.fixes.MaskedArray = MaskedArray

import skopt
from sklearn import metrics
from sklearn import ensemble
from sklearn import model_selection
from skopt import gp_minimize
from skopt import space

def optimize(params, param_names, x, y):
    """
    The main optimizatiion function
    This function takes all the arguments from the search space and training features and target.
    It then initialized the models by setting the chosen paramaters and runs crossvalidation and returns a 
    negative accuracy score
    :param params: list of params from gp_minimize
    :param param_names: list of param names. Order is important!
    :param x: training data
    :param y: labels/targets
    :return negative accuracy after 5 folds
    """

    # convert params to dictionary
    params = dict(zip(param_names, params))

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
    param_space = [
        # max_depth is an integer between 3 and 10
        space.Integer(3, 15, name="max_depth"),
        # n_estimators is an interger between 100 and 1500
        space.Integer(100, 1500, name="n_estimators"),
        # criterion is a category. List of categories
        space.Categorical(["gini", "entropy"], name="criterion"),
        # real numbered space and definf a distribution to sample from
        space.Real(0.01, 1, prior="uniform", name="max_features")
    ]

    # make a list of param names,
    ## same order as the search space
    param_names = ["max_depth", "n_estimators", "criterion", "max_features"]

    # by using functools - partial, creating a new function which has same parameters
    #   as the optimize function except for the fact that only one param i.e. "params" parameter
    #  is required
    # this is how gp_minimize expects the optimization function to be.
    # define optimization function here
    optimization_function = partial(optimize, param_names=param_names,x = X, y=y)

    # now we call gp_minimize from scikit-optimize
    # gp_minimize used bayesian optimization for minimization pf the optimization function
    # we need a space of paramters, the function itself, the number of iterations we want to have
    result = gp_minimize(
        optimization_function, 
        dimensions = param_space,
        n_calls =15,
        n_random_starts=10,
        verbose =10)

    # create best params and print it
    best_params = dict(zip(param_names, result.x))

    print(best_params)

# notebook part
# from skopt.plots import plot_convergence
# plot_convergence(result)