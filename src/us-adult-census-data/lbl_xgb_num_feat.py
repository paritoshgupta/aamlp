import pandas as pd
import config
import itertools
import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing

def feature_engineering(df, cat_cols):
    """
    Creates 2 combinations of values
    """
    # list(itertools.combinations([1,2,3], 2)) will return - [(1,2), (1,3), (2,3)]

    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[:, c1+"_" + c2] = df[c1].astype(str) + "_" + df[c2].astype(str)

    return df

def run(fold):
    # read training data
    df = pd.read_csv(config.TRAINING_FILE_WITH_FOLDS)

    # numerical columns
    num_cols = ["fnlwgt", "age", "capital.gain", "capital.loss", "hours.per.week"]

    # map targets to 0 and 1
    target_mapping = {"<=50K": 0, ">50K": 1}
    df.loc[:, "income"] = df.income.map(target_mapping)

    # list of categorical columns for feature engineering
    cat_cols = [c for c in df.columns if c not in ["kfold", "income"] and c not in num_cols]

    # feature engineering
    df = feature_engineering(df, cat_cols)

    # all columns except kfold, income
    features = [f for f in df.columns if f not in ("kfold", "income")]

    # fill NAs with NONE and don't encode numerical columns
    for col in features:
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # fill NAs with NONE
    for col in features:
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df.loc[:, col] = lbl.transform(df[col])

    # get training and validatiom using folds
    df_train = df[df.kfold != 0].reset_index(drop=True)
    df_valid = df[df.kfold ==0].reset_index(drop=True)

    # get training and validation
    x_train = df_train[features].values
    x_valid = df_valid[features].values

    # initialize xgb model
    model = xgb.XGBClassifier(n_jobs=-1)

    # fit the model
    model.fit(x_train, df_train.income.values)

    # predict
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # auc 
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)
    
