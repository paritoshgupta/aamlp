# import pandas and model_selection module of scikit-learn

import pandas  as pd
from sklearn import model_selection

if __name__ =="__main__":

    # Read training data
    df = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")

    # create a new column kfold and fill it with -1
    df["kfold"] = -1

    # randomize the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch labels
    y = df.target.values

    # initiate the kfold class from model_selection module
    skf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for fold, (train, valid) in enumerate(skf.split(X=df, y=y)):
        df.loc[valid, 'kfold'] = fold

    df.to_csv("../input/cat-in-the-dat-ii/train_folds.csv", index=False)


