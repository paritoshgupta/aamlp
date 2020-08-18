import copy
import pandas as pd
import config
import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing

def mean_target_encoding(data):

    # make a copy of dataframe
    df = copy.deepcopy(data)

    # list of numerical columns
    num_cols = ["fnlwgt", "age", "capital.gain", "capital.loss", "hours.per.week"]

    # map targets to 0s and 1s
    mapping = {"<=50K": 0, ">50K": 1}
    df.loc[:, "income"] = df.income.map(mapping)

    # all columns are features except numerical columns and kfold columns
    features = [f for f  in df.columns if f not in ("kfold", "income") and f not in num_cols]

    # fill NAs with NONE
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # label encode the columns
    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df.loc[:, col] = lbl.transform(df[col])

    # list to store validation dataframes
    encoded_dfs = []

    # go over all folds
    for fold in range(5):
        # fetch training and validation data
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        # for all categorical columns
        for column in features:
            # create dict of category: mean target
            mapping_dict = dict(df_train.groupby(column)["income"].mean())

            # column_enc is the new column for mean encoding
            df_valid.loc[:, column+"_enc"] = df_valid[column].map(mapping_dict)

        encoded_dfs.append(df_valid)

    # create full data frame again and return
    encoded_df = pd.concat(encoded_dfs, axis=0)

    return encoded_df

def run(df, fold):

    # get training and validation data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # features
    features = [f for f in df.columns if f not in ("kfold", "income")]

    # training and validation data
    x_train = df_train[features].values
    x_valid = df_valid[features].values

    # initialize xgb model
    model = xgb.XGBClassifier(n_jobs=-1, max_depth=7)

    # fit on train data
    model.fit(x_train, df_train.income.values)

    # predict
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # auc
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":

    df = pd.read_csv(config.TRAINING_FILE_WITH_FOLDS)

    # create mean target encoded categories and munge data
    df = mean_target_encoding(df)

    # run training and validation
    for fold_ in range(5):
        run(df, fold_)



