#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
import csv

# Classifier training
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from imblearn.over_sampling import SMOTE

# Cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV, RFE

def train_classifier(classifier, strategy, oversample, X_train, y_train):
    """Computes a multiclass classification.

    Args:
        classifier: An instance of a scikit-learn classifier.
        strategy: A string defining which strategies will be used for training.
        oversample: A boolean variable that defines if SMOTE oversampling should be applied or not.
        X_train: A matrix containing features for training.
        X_test: A matrix containing features for testing.
        y_train: A column containing labels for training.
        y_test: A column containing labels for testing.

    Returns:
        A classification model and its performance report
    """

    if strategy == 'one_vs_rest':
        model = OneVsRestClassifier(classifier)
    if strategy == 'one_vs_one':
        model = OneVsOneClassifier(classifier)

    if oversample:
        X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
        model.fit(X_resampled, y_resampled)
    else:
        model.fit(X_train, y_train)

    return model

def features_cross_validation(classifier, strategy, oversample, X_train, y_train, results_dir):
    print("Running feature selection")

    if strategy == 'one_vs_rest':
        model = OneVsRestClassifier(classifier)
    if strategy == 'one_vs_one':
        model = OneVsOneClassifier(classifier)

    cv = StratifiedKFold(n_splits=2)

    if oversample:
        X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)

        with open(os.path.join(results_dir, 'usefulness_of_features.txt'), 'w') as usefulness_file:
            # selector = RFECV(model, cv=cv, n_jobs=-1, min_features_to_select=10)
            selector = RFE(model, n_features_to_select=10)
            selector = selector.fit(X_resampled, y_resampled)
            print(selector.ranking_)
            print(selector.n_features_in_)
            usefulness_file.write("Usefulness of features:\n")
            usefulness_file.write(str(selector.feature_names_in_))
            usefulness_file.write(str(selector.ranking_))
    else:
        with open(os.path.join(results_dir, 'usefulness_of_features.txt'), 'w') as usefulness_file:
            # selector = RFECV(model, cv=cv, n_jobs=-1, min_features_to_select=10)
            selector = RFE(model, n_features_to_select=10)
            selector = selector.fit(X_train, y_train)
            print(selector.ranking_)
            print(selector.n_features_in_)
            usefulness_file.write("Usefulness of features:\n")
            usefulness_file.write(str(selector.n_features_in_))
            usefulness_file.write(str(selector.ranking_))
