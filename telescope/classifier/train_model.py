#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import pandas
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score

def train(classifier, X_train, X_test, y_train, y_test, strategy):
    """Computes a multi-label classification.

    This approach is used by `one-vs-the-rest`, `classifier-chains`, and
    `label-powerset` strategies. For each classifier, the classes are fitted
    at the same time or in sequence. Since all the classes are represented by one
    and only one classifier, it is possible to gain knowledge about the classes
    by inspecting this unique classifier.

    Args:
        classifier: An instance of a scikit-learn classifier.
        classes: A list of strings representing the classes to be trained.
        X_train: A matrix containing features for training.
        y_train: A one-column dataframe containing labels for training.
        strategy: A string defining which of the three strategies will be used.

    Returns:
        A classification model and its performance report
    """
    if strategy == 'one-vs-the-rest':
        model = OneVsRestClassifier(classifier)
    if strategy == 'classifier-chains':
        model = ClassifierChain(classifier)
    if strategy == 'label-powerset':
        model = LabelPowerset(classifier)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=y_train.columns)

    return model, report
