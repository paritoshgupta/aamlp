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
        "n_estimators": [100, 200, 250, 300, 400, 500],
        "max_depth": [1,2,5, 7, 11, 15],
        "criterion": ["gini", "entropy"]
    }

    # initialize grid search
    # cv = 5, we are using 5 fold cv (not stratified)

    model = model_selection.GridSearchCV(
        estimator=classifier,
        param_grid = param_grid,
        scoring = "accuracy",
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

    