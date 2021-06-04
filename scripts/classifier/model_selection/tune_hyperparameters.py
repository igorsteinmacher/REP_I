#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def tune_hyperparameters(classifier, hyperparameters, strategy, oversample, X_train, y_train):
    pipeline_args = []

    if oversample:
        oversample = ('smt', SMOTE())
        pipeline_args.append(oversample)

    if strategy == 'one_vs_rest':
        strategy = ('clf', OneVsRestClassifier(classifier))

    if strategy == 'one_vs_one':
        strategy = ('clf', OneVsOneClassifier(classifier))

    pipeline_args.append(strategy)
    pipeline = Pipeline(pipeline_args)
    cross_validation = StratifiedKFold(n_splits=10)

    model = GridSearchCV(pipeline, hyperparameters, scoring="f1_weighted", cv=cross_validation)
    model.fit(X_train, y_train)

    return model