import pandas as pd
import config
from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb

def run(fold):

    # read training data
    df = pd.read_csv(config.TRAINING_FILE_WITH_FOLDS)

    # extract features
    features = [f for f in df.columns if f not in ["id", "target", "kfold"]]

    # fill NA values with NONE
    for col in features:
        # le = preprocessing.LabelEncoder()
        df.loc[:, col] = df[col].astype('str').fillna("NONE")

    # Label encoder
    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df.loc[:, col] = lbl.transform(df[col])

    # get training data
    df_train = df[df.kfold !=0].reset_index(drop=True)

    # get validation data
    df_valid = df[df.kfold ==0].reset_index(drop=True)

    # get training data
    x_train = df_train[features]

    # get validation data
    x_valid = df_valid[features]

    # initialize xgb model 
    model = xgb.XGBClassifier(n_jobs=-1, max_depth=7, estimators=200)

    # fit model on training data
    model.fit(x_train, df_train.target.values)

    # predict on validation data
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc} ")

if __name__=="__main__":
    for fold_ in range(5):
        run(fold_)


