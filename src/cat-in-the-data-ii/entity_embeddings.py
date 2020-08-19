import os
import config
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils

def create_model(data, catcols):
    """
    This function returns a compiled tf.keras model for entity embeddings
    :param data: pandas dataframe
    :param catcols: list of categorical columns
    :return: compiled tf.keras models
    """

    # initialize list of inputs for embeddings
    inputs = []

    # initialize list of outputs for embeddings
    outputs = []

    # loop over all categorical columns:
    for c in catcols:
        # number of unique values in column
        num_unique_values = int(data[c].nunique())
        # simple dimension of embedding calculator
        # min size: size is half of unique values
        # max size is 50. depends on number of unique categories. 
        embed_dim = int(min(np.ceil((num_unique_values)/2), 50))

        # simple keras input layer with size 1
        inp = layers.Input(shape = (1, ))

        # add embedding layer to raw input
        # embedding size is always 1 more than unique values in input
        out = layers.Embedding(num_unique_values+1, embed_dim, name=c)(inp)

        # 1-d spatial dropout is the standard for embedding layers
        # we can use it in NLP tasks too
        out = layers.SpatialDropout1D(0.3)(out)

        # reshape the input to the dimension of embedding
        out = layers.Reshape(target_shape = (embed_dim, ))(out)

        # add input to input list
        inputs.append(inp)

        # add outputs to output list
        outputs.append(out)

    # concatenate all output layers
    x = layers.Concatenate()(outputs)

    # add a batchnorm layer
    # from here, everything is upto you
    # try different architectures, this works quite good
    # if you have numerical featurs, add them here or in concatenate layer
    x = layers.BatchNormalization()(x)

    # a bunch of dense layers with droput
    x = layers.Dense(300, activation = "relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation = "relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    # using softmax as it is a two class problem
    y = layers.Dense(2, activation="softmax")(x)

    # create final model
    model = Model(inputs=inputs, outputs=y)

    # compile the model
    # we use adam and binary cross entropy
    model.compile(loss="binary_crossentropy", optimizer='adam')
    return model

def run(fold):

    print(f"Starting Fold --> {fold}")
    # read training data
    df = pd.read_csv(config.TRAINING_FILE_WITH_FOLDS)

    # extract columns
    features = [f for f in df.columns if f not in ("id", "target", "kfold")]

    # fill NAs with NONE 
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # encode all features with LabelEncoder
    for feat in features:
        lbl_enc = preprocessing.LabelEncoder()
        df.loc[:, feat] = lbl_enc.fit_transform(df[feat].values)

    # get training and validation data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold ==fold].reset_index(drop=True)

    # create tf.keras model
    model = create_model(df, features)

    # our features are list of lists
    x_train = [df_train[features].values[:, k] for k in range(len(features))]
    x_valid = [df_valid[features].values[:, k] for k in range(len(features))]

    # get target columns
    y_train = df_train.target.values
    y_valid = df_valid.target.values
    
    # convert target columns to categories 
    # just binarization
    y_train_cat = utils.to_categorical(y_train)
    y_valid_cat = utils.to_categorical(y_valid)

    # fit the model
    model.fit(x_train, y_train_cat,
                validation_data = (x_valid, y_valid_cat),
                verbose=10,
                batch_size=1024,
                epochs=3)

    # generate validation predictions
    valid_preds = model.predict(x_valid)[:, 1]

    # print roc auc score
    auc = metrics.roc_auc_score(y_valid, valid_preds)
    print(f"Fold = {fold}, AUC = {auc}")

    # clear session 
    K.clear_session()

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)


