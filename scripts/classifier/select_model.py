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


def get_models_performance(classifiers, hyperparameters, strategies, oversample, X_train, y_train):
    models_performance = {}

    for classifier, hyperparameters in zip(classifiers, hyperparameters):
        classifier_name = type(classifier).__name__
        print("Classifier: " + classifier_name)

        for strategy in strategies:
            print("Multiclass strategy: " + strategy )

            for oversample_condition in oversample:
                print("Oversampling: " + str(oversample_condition).lower())

                training_args = {
                    'classifier': classifier,
                    'hyperparameters': hyperparameters,
                    'strategy': strategy,
                    'oversample': oversample_condition,
                    'X': X_train,
                    'y': y_train
                }

                f1_scores_mean = nested_cross_validation(**training_args)
                models_performance[training_args] = f1_scores_mean
    
    return models_performance

def nested_cross_validation(classifier, hyperparameters, strategy, oversample, X_train, y_train):
    """Computes a multiclass nested ten-fold cross-validation.

    To better evaluate how each algorithm performs in our data,
    a nested stratified k-folding cross-validation approach was applied.

    Args:
        classifier: An instance of a scikit-learn classifier.
        hyperparameters: Hyperparameters to be used in Gridsearch.
        strategy: A string defining which strategies will be used for training.
        oversample: A boolean variable that defines if SMOTE oversampling is applied.
        X: A matrix containing features.
        y: A column containing labels.

    Returns:
        Mean of F1 scores from the nested cross-validation.
    """
    pipeline_args = []

    if oversample:
        oversample = ('smt', SMOTE())
        pipeline_args.append(oversample)

    if strategy == 'one-vs-rest':
        strategy = ('clf', OneVsRestClassifier(classifier))

    if strategy == 'one-vs-one':
        strategy = ('clf', OneVsOneClassifier(classifier))

    pipeline_args.append(strategy)
    pipeline = Pipeline(pipeline_args)

    inner_cv = StratifiedKFold(n_splits=10)
    outer_cv = StratifiedKFold(n_splits=10)

    inner_estimator = GridSearchCV(pipeline, hyperparameters, cv=inner_cv)
    outer_results = cross_val_score(inner_estimator, X_train, y_train, scoring='f1_weighted', cv=outer_cv)

    return outer_results.mean()

def hyperparameters_tuning(classifier, hyperparameters, strategy, oversample, X_train, y_train):
    pipeline_args = []

    if oversample:
        oversample = ('smt', SMOTE())
        pipeline_args.append(oversample)

    if strategy == 'one-vs-rest':
        strategy = ('clf', OneVsRestClassifier(classifier))

    if strategy == 'one-vs-one':
        strategy = ('clf', OneVsOneClassifier(classifier))

    pipeline_args.append(strategy)
    pipeline = Pipeline(pipeline_args)

    cross_validation = StratifiedKFold(n_splits=10)
    model = GridSearchCV(classifier, hyperparameters, scoring="f1_weighted", cv=cross_validation)
    model.fit(X_train, y_train)

    return model.best_params_