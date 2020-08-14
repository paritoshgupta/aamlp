import argparse
import os
import config
import model_dispatcher

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree


def run(fold, model):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # training data is where the kfold is not equal to provided fold
    df_train = df[df.kfold !=fold].reset_index(drop=True)

    # validation data is where the kfold is equal to provided fold
    df_valid = df[df.kfold ==fold].reset_index(drop=True)

    # drop the label column from dataframe and convert it to a numpy array by using .values
    # target is label column in the dataframe
    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values

    # similarly for validation, we have
    x_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values

    # initialize simple decision tree classifier from sklearn

    clf = model_dispatcher.models[model]
    #clf = tree.DecisionTreeClassifier()

    # fit the model on training data
    clf.fit(x_train, y_train)

    # create predictions for validation samples
    preds = clf.predict(x_valid)

    # calculate & print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy ={accuracy}")
    # fold number and accuracy
    
    # save the model
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin"))


if __name__=="__main__":

    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the differenr arguments you need and their type
    parser.add_argument("--fold", type = int)

    parser.add_argument("--model", type=str)
    # read the arguments from the command line
    args = parser.parse_args()

    # run the fold specified by command line arguments
    run(fold=args.fold, model=args.model)
