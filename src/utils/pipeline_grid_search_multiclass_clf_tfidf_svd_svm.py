import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def quadratic_weighted_kappa(y_true, y_pred):
    # create a wrapper for cohen's kappa with quadratic weights"

    return metrics.cohen_kappa_score(y_true, y_pred, weights = "quadratic")

if __name__ == "__main__":

    # Load the training data
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")


    # drop id columns from train and test
    idx = test.id.values.astype(int)
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)

    # create labels, drop useless columns here
    y = train.relevance.values

    # do lambda magic on text ccolumns
    traindata = list(train.apply(lambda x: '%s %s' % (x['text1'], x['text2'], axis=1))
    testdata = list(test.apply(lambda x: '%s %s' % (x['text1'], x['text2'], axis=1))
    
    # tfidf vectrozier
    tfv = TfidfVectorizer(min_df=3,
    max_features=None,
    strip_accents='unicode',
    analyzer='word',
    token_pattern = r'\w{1,}'),
    ngram_range = (1,3),
    use_idf=1,
    smooth_idf=1,
    sublinear_tf=1,
    stop_words='english'
    )

    # fit TFIDF
    tfv.fit(traindata)
    X = tfv.transform(traindata)
    X_test = tfv.transform(testdata)

    # initialize SVD
    svd = TruncatedSVD()

    # initialize the standard scaler
    scl = StandardScaler()

    # using SVM here
    svm_model = SVC()

    # create the pipeline
    clf = pipeline.Pipeline([('svd', svd), ('scl': scl), ('svm': svm_model)])

    # create a parameter grid to search for best parameters in the pipeline
    param_grid = {'svd__n_components': [200, 300],
                  'svm__C': [10, 12]}

    # kappa scorer
    kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa,
                                        greater_is_better=True)

    # initialize the grid search model
    model = model_selection.GridSearchCV(
        estimator = clf,
        param_grid = param_grid,
        scoring = kappa_scorer,
        verbose = 10,
        n_jobs = -1,
        refit=True,
        cv=5
    )                                        

    # fit grid search model
    model.fit(X,y)
    print(f"Best Score -> {model.best_score_")
    print(f"Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"{param_name} --> {best_parameters[param_name]")

    # get best model
    best_model = model.best_estimator_
    # Fit model with best parameters optimized for QWK
    best_model.fit(X,y)
    preds = model.predict(X_test)