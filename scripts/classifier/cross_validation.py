#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score, precision_score, f1_score
from imblearn.pipeline import Pipeline, make_pipeline


def k_fold_cross_validation(classifier, strategy, oversample, X, y):
    """Computes a multiclass cross-validation.

    To better evaluate how each algorithm performs in our data,
    a stratified k-folding cross-validation approach was applied
    in our study.

    Args:
        classifier: An instance of a scikit-learn classifier.
        strategy: A string defining which strategies will be used for training.
        oversample: A boolean variable that defines if SMOTE oversampling should be
         applied or not.
        X: A matrix containing features.
        y: A column containing labels.

    Returns:
        An array containing the F1 scores for the k folds
    """
    pipeline_params = []

    if oversample:
        oversampling = SMOTE()
        pipeline_params.append(('smt', oversampling))

    if strategy == 'one-vs-rest':
        model = OneVsRestClassifier(classifier)

    if strategy == 'one-vs-one':
        model = OneVsOneClassifier(classifier)

    pipeline_params.append(('clf', model))
    pipeline = Pipeline(pipeline_params)

    k_splits = 10
    k_fold = StratifiedKFold(n_splits=k_splits, shuffle=True)          
    results = cross_val_score(pipeline, X, y, scoring='f1_weighted', cv=k_fold, n_jobs=-1)
    
    return results