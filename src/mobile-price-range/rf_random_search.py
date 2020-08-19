import numpy as np
import pandas as pd
import config
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

if __name__ == "__main__":
    # read the training data
    df = pd.read_csv(config.TRAINING_FILE)

    # predictors
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    # define model here, n_jobs=-1 => use all cores
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)

    # define a grid of parameters (dictionaries)
    param_grid = {
        "n_estimators": np.arange(100, 1500, 100),
        "max_depth": np.arange(1,31),
        "criterion": ["gini", "entropy"]
    }

    # initialize grid search
    # cv = 5, we are using 5 fold cv (not stratified)

    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions = param_grid,
        scoring = "accuracy",
        n_iter = 20,
        verbose = 10,
        n_jobs =-1,
        cv = 5)

    # fit the model
    model.fit(X,y)
    print(f"Best Score: {model.best_score_}")
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")

    