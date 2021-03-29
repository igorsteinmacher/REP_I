#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def train(classifier, strategy, oversample, categories, X_train, X_test, y_train, y_test):
    """Computes a multiclass classification.

    For each classifier, the classes are fitted at the same time or in sequence. 
    Since all the classes are represented by one and only one classifier,
    it is possible to gain knowledge about the classes by inspecting this unique classifier.

    Args:
        classifier: An instance of a scikit-learn classifier.
        strategy: A string defining which strategies will be used for training.
        oversample: A boolean defining if SMOTE oversampling should be applied or not.
        X_train: A matrix containing features for training.
        X_test: A matrix containing features for testing.
        y_train: A column containing labels for training.
        y_test: A column containing labels for testing.

    Returns:
        A classification model and its performance report
    """
    if strategy == 'one-vs-rest':
        model = OneVsRestClassifier(classifier)
    if strategy == 'one-vs-one':
        model = OneVsOneClassifier(classifier)

    if oversample:
        X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
        model.fit(X_resampled, y_resampled)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=categories)

    return model, report
