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


def evaluate_estimators_performance(classifiers, hyperparameters, strategies,
                                    oversample, X_train, y_train):
                                    
    estimators_performance = {}
    best_estimator = {'identifier': None, 'f1_mean': 0.0, 'args': None}

    for classifier, hyperparameters in zip(classifiers, hyperparameters):
        classifier_name = type(classifier).__name__
        print("Classifier: " + classifier_name)

        for strategy in strategies:
            print("Multiclass strategy: " + strategy )

            for oversample_condition in oversample:
                oversample_name = str(oversample_condition).lower()
                print("Oversampling: " + oversample_name)

                training_args = {
                    'classifier': classifier,
                    'hyperparameters': hyperparameters,
                    'strategy': strategy,
                    'oversample': oversample_condition,
                    'X_train': X_train,
                    'y_train': y_train
                }

                f1_mean = nested_cross_validation(**training_args)
                
                estimator_identifier = 'estimator_' + classifier_name + '_strategy_'\
                                       + strategy + '_smote_' + oversample_name

                estimators_performance[estimator_identifier] = {'f1_mean': f1_mean,
                                                                'args': training_args}
                
                if f1_mean > best_estimator['f1_mean']:
                    best_estimator['estimator'] = estimator_identifier
                    best_estimator['f1_mean'] = f1_mean
                    best_estimator['args'] = training_args
    
    return best_estimator, estimators_performance

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

    if strategy == 'one_vs_rest':
        strategy = ('clf', OneVsRestClassifier(classifier))

    if strategy == 'one_vs_one':
        strategy = ('clf', OneVsOneClassifier(classifier))

    pipeline_args.append(strategy)
    pipeline = Pipeline(pipeline_args)

    inner_cv = StratifiedKFold(n_splits=10)
    outer_cv = StratifiedKFold(n_splits=10)

    inner_estimator = GridSearchCV(pipeline, hyperparameters, cv=inner_cv)
    outer_results = cross_val_score(inner_estimator, X_train, y_train, scoring='f1_weighted', cv=outer_cv)

    return outer_results.mean()
