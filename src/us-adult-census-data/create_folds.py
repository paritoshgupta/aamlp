# import pandas and model_selection module of scikit-learn

import pandas  as pd
from sklearn import model_selection
import config

if __name__ =="__main__":

    # Read training data
    print(f"Reading input data...")
    df = pd.read_csv(config.TRAINING_FILE)

    # create a new column kfold and fill it with -1
    df["kfold"] = -1

    # randomize the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch labels
    y = df.income.values

    # initiate the kfold class from model_selection module
    skf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for fold, (train, valid) in enumerate(skf.split(X=df, y=y)):
        df.loc[valid, 'kfold'] = fold

    print(f"Writing input data after adding fold column...")
    df.to_csv(config.TRAINING_FILE_WITH_FOLDS, index=False)


