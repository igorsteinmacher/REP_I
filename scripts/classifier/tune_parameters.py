#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

from prepare_data import shuffle_and_split
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from imblearn.over_sampling import SMOTE

def hyperparameters_tuning(classifier, strategy, oversample, X, y):
    classifier_name = type(classifier).__name__

    if classifier_name == 'LinearSVC':
        X_train, X_test, y_train, y_test = shuffle_and_split(X, y)
        param_grid = [{'max_iter': [1000, 10000, 50000],
                       'tol': [1e-3, 1e-4, 1e-5],
                       'C': [1, 10, 100, 1000]}]

        if strategy == 'one-vs-rest':
            model = OneVsRestClassifier(classifier)
        if strategy == 'one-vs-one':
            model = OneVsOneClassifier(classifier)

        model = GridSearchCV(classifier, param_grid, scoring="f1_weighted")

        if oversample:
            X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
            model.fit(X_resampled, y_resampled)
        else:
            model.fit(X_train, y_train)

        return model.best_params_

    else:
        print("Please, define the set of parameters in tune_parameters.py")
