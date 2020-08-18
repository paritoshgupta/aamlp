import pandas as pd
import config
import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing

def run(fold):

    # read training data
    df = pd.read_csv(config.TRAINING_FILE_WITH_FOLDS)

    # list of numerical columns
    num_cols = ["fnlwgt", "age", "capital.gain", "capital.loss", "hours.per.week"]

    # drop numerical columns
    df = df.drop(num_cols, axis=1)

    # map targets to 0 and 1
    target_mapping = {"<=50K": 0, ">50K": 1}
    df.loc[:, "income"] = df.income.map(target_mapping)

    # list of relevant features
    features = [f for f in df.columns if f not in ["kfold", "income"]]

    # fill NAs with NONE
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # label encode
    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df.loc[:, col] = lbl.transform(df[col])

    # get training and validation using folds
    df_train = df[df.kfold !=fold].reset_index(drop=True)
    df_valid = df[df.kfold ==fold].reset_index(drop=True)

    # get training and validation data
    x_train = df_train[features].values
    x_valid = df_valid[features].values

    # initialize xgb model
    model = xgb.XGBClassifier(n_jobs=-1)

    # fit model on training data
    model.fit(x_train, df_train.income.values)

    # predict on validation data
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # auc
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)