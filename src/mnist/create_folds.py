import pandas as pd
from sklearn import model_selection
# import config

if __name__ =="__main__":
    # Read training data
    print(f"Reading input data...")
    df = pd.read_csv("../input/mnist/mnist_train.csv")
    # df = pd.read_csv(config.RAW_FILE)
    
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # initialize the kfold class from model_selection module
    kf = model_selection.KFold(n_splits=5)

    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, "kfold"] = fold

    # save the new csv with kfold column
    print(f"Writing input data after adding fold column...")
    df.to_csv("../input/mnist/mnist_train_folds.csv", index=False)
    # df.to_csv(config.TRAINING_FILE, index=False)
    
