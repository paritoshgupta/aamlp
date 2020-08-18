import pandas as pd
import config

from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model


def run(fold):
    # load the full training data
    df = pd.read_csv(config.TRAINING_FILE_WITH_FOLDS)
    
    # list of numerical columns
    num_cols = ["fnlwgt", "age", "capital.gain", "capital.loss", "hours.per.week"]

    # drop numerical columns
    df = df.drop(num_cols, axis=1)

    # map targets to 0 and 1
    target_mapping = {"<=50K": 0, ">50K": 1}
    df.loc[:, "income"] = df.income.map(target_mapping)

    # features except income and kfold column
    features = [f for f in df.columns if f not in ["kfold", "income"]]

    # fill all NaN with NONE
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # get training data
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # initialize one hot encoder
    ohe = preprocessing.OneHotEncoder()

    # fit on full data
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
    ohe.fit(full_data[features])

    # transform on train and validation
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    # initialize logistoc regression model
    model = linear_model.LogisticRegression()

    # fit model on training data
    model.fit(x_train, df_train.income.values)

    # predict on validation data
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get auc 
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, AUC = {auc}")

if __name__=="__main__":
    for fold_ in range(5):
        run(fold_)