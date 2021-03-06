#Classification using logostic regression

import os
import timeit
import numpy as np
import glob
from collections import defaultdict
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from utils import GENRE_LIST, GENRE_DIR, CHART_DIR
from fft_generator import read_fftx, plot_confusion_matrix
from matplotlib import pylab

genre_list = GENRE_LIST

def train_model(X, Y, name, plot=False):
    """
        train_model(vector, vector, name[, plot=False])
        
        Trains and saves model to disk.
    """
    labels = np.unique(Y)

    cv = ShuffleSplit(n=len(X), test_size=0.3, random_state=0)

    train_errors = []
    test_errors = []

    scores = []
    pr_scores = defaultdict(list)
    precisions, recalls, thresholds = defaultdict(list), defaultdict(list), defaultdict(list)

    clfs = []  # for the median

    cms = []
    
    for train, test in cv:
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]

        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        clfs.append(clf)

        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        scores.append(test_score)

        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)

        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cms.append(cm)

    #save the trained model to disk
    joblib.dump(clf, 'C:\\Users\\hp\\Desktop\\project\\logregdata.rar')

    
    return np.mean(train_errors), np.mean(test_errors), np.asarray(cms)


if __name__ == "__main__":
    start = timeit.default_timer()
    print (" Starting classification \n")
    print (" Classification running ... \n") 
    X, y = read_fftx(genre_list)
    train_avg, test_avg, cms = train_model(X, y, "fftx", plot=True)
    cm_avg = np.mean(cms, axis=0)
    cm_norm = cm_avg / np.sum(cm_avg, axis=0)
    print (" Classification finished \n")
    stop = timeit.default_timer()
    print (" Total time taken (s) = ", (stop - start))
    print ("\n Plotting confusion matrix ... \n")
    plot_confusion_matrix(cm_norm, genre_list, "fftx","FFTX classifier - Confusion matrix")
    print (" All Done\n")
