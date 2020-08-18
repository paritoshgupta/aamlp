import pandas as pd
import config
from sklearn import preprocessing
from sklearn import metrics
from sklearn import ensemble

def run(fold):
    
    print(f"########## Running for fold --> {fold} ##########")

    # load the training data
    print(f"Reading training data with folds...")
    df = pd.read_csv(config.TRAINING_FILE_WITH_FOLDS)

    # all columns except id, target and kfold columns
    # print(f"Selecting input features")
    features = [f for f in df.columns if f not in ["id", "target", "kfold"]]

    # fill all NaN values with NONE
    # note all columns are being converted to strings as it doesn't matter because all are catgeries
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    
    # label encoding one column at a time
    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df.loc[:, col] = lbl.transform(df[col])

    
    # training data using folds
    print(f"Getting train data..")
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    # validation data using folds
    print(f"Getting validation data..")
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].values
    
    # get validation data
    x_valid = df_valid[features].values

    # initialize logistic regression model
    model = ensemble.RandomForestClassifier(n_jobs=-1)
    
    # fit model on training data
    print(f"Fitting Random Forest model..")
    model.fit(x_train, df_train.target.values)

    # predict on validation data
    # we need the probability values as we are calculating AUC
    # Get the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get auc roc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f"AUC Score for fold - {fold} --> {auc}")

if __name__ =="__main__":
    # run for any fold
    for fold_ in range(5):
        run(fold_)