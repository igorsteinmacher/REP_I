#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
from imblearn.pipeline import Pipeline

# Cross-validation
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV

# Oversampling techniques
from imblearn.over_sampling import SMOTE

# Training strategies
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

# Classification Algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def evaluate_estimators_performance(classifiers, strategies, oversample, 
                                    X_train, y_train, results_dir):

    classifiers_available = {
        'rf': RandomForestClassifier(),
        'svc': LinearSVC(),
        'mnb': MultinomialNB(),
        'knn': KNeighborsClassifier(),
        'lr': LogisticRegression(),
    }

    hyperparameters_available = {
        'rf': {'clf__estimator__max_depth': [None, 50, 100],
               'clf__estimator__n_estimators': [50, 100, 150]},
        'svc': {'clf__estimator__tol': [1e-3, 1e-4, 1e-5],
                'clf__estimator__C': [0.5, 1, 1.5],
                'clf__estimator__max_iter': [500, 1000, 1500]},
        'mnb': {'clf__estimator__alpha': [0.5, 1, 1.5],
                'clf__estimator__fit_prior': [True, False]},
        'knn': {'clf__estimator__n_neighbors': [3, 5, 7]},
        'lr': {'clf__estimator__tol': [1e-3, 1e-4, 1e-5],
               'clf__estimator__C': [0.5, 1, 1.5],
               'clf__estimator__max_iter': [50, 100, 150]},
    }

    strategies_available = {
        'ovr': 'one_vs_rest',
        'ovo': 'one_vs_one'
    }

    selected_classifiers = [classifiers_available[classifier]
                            for classifier in classifiers]

    selected_hyperparameters = [hyperparameters_available[classifier]
                                for classifier in classifiers]

    selected_strategies = [strategies_available[strategy]
                            for strategy in strategies]

    for classifier, hyperparameters in zip(selected_classifiers, selected_hyperparameters):
        classifier_name = type(classifier).__name__
        print("Classifier: " + classifier_name)

        for strategy in selected_strategies:
            print("Multiclass strategy: " + strategy)

            for oversample_condition in oversample:
                oversample_bool = str(oversample_condition).lower()
                print("Oversampling: " + oversample_bool)

                identifier = 'estimator_' + classifier_name + '_strategy_'\
                             + strategy + '_smote_' + oversample_bool

                training_args = {
                    'classifier': classifier,
                    'identifier': identifier,
                    'hyperparameters': hyperparameters,
                    'strategy': strategy,
                    'oversample': oversample_condition,
                    'X_train': X_train,
                    'y_train': y_train
                }

                nested_cross_validation(**training_args, results_dir=results_dir)
    
def nested_cross_validation(classifier, identifier, hyperparameters, strategy, 
                            oversample, X_train, y_train, results_dir):
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

    performance_filepath = os.path.join(results_dir, identifier)
    
    with open(performance_filepath, 'w') as performance_file:
        cross_val_score_mean = outer_results.mean()
        performance_file.write('F1 Weighted (Mean): ' + str(cross_val_score_mean) + '\n')

        inner_estimator.fit(X_train, y_train)
        performance_file.write('Best parameters: ' + str(inner_estimator.best_params_) + '\n')
        performance_file.write('Best estimator: ' + str(inner_estimator.best_estimator_) + '\n')
