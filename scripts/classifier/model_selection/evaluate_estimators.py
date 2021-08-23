#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
import numpy as np
np.set_printoptions(threshold=np.inf)
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
from sklearn.dummy import DummyClassifier

def evaluate_estimators_performance(classifiers, strategies, oversample, 
                                    X_train, y_train, results_dir):

    classifiers_available = {
        'rf': RandomForestClassifier(),
        'svc': LinearSVC(),
        'mnb': MultinomialNB(),
        'knn': KNeighborsClassifier(),
        'lr': LogisticRegression(),
        'dmr': DummyClassifier(strategy='uniform', random_state=42),
        'dmf': DummyClassifier(strategy='most_frequent')
    }

    hyperparameters_available = {
        'rf': {'clf__estimator__n_estimators': [75, 100, 125]},
        'svc': {'clf__estimator__tol': [1e-3, 1e-4, 1e-5],
                'clf__estimator__C': [0.5, 1, 1.5],
                'clf__estimator__max_iter': [500, 1000, 1500]},
        'mnb': {'clf__estimator__alpha': [0.5, 1, 1.5],
                'clf__estimator__fit_prior': [True, False]},
        'knn': {'clf__estimator__n_neighbors': [3, 5, 7]},
        'lr': {'clf__estimator__tol': [1e-3, 1e-4, 1e-5],
               'clf__estimator__C': [0.5, 1, 1.5],
               'clf__estimator__max_iter': [50, 100, 150]},
        'dmr': {}, # This is a dummy classifier
        'dmf': {}  # This is a dummy classifier
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
        print("Evaluating estimator: " + classifier_name)

        for strategy in selected_strategies:
            print("Multiclass strategy: " + strategy)

            for oversample_condition in oversample:
                oversample_bool = str(oversample_condition).lower()
                print("Oversample: " + oversample_bool)

                estimator_args = {
                    'classifier': classifier,
                    'hyperparameters': hyperparameters,
                    'strategy': strategy,
                    'oversample': oversample_condition,
                    'X_train': X_train,
                    'y_train': y_train
                }

                internal_evaluation, f1_weighted_scores_means = nested_cross_validation(**estimator_args)

                export_estimator_results(estimator_args, internal_evaluation, 
                                         f1_weighted_scores_means, results_dir)
    
def nested_cross_validation(classifier, hyperparameters, strategy, 
                            oversample, X_train, y_train):

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
        pipeline_args.append(strategy)
    elif strategy == 'one_vs_one':
        strategy = ('clf', OneVsOneClassifier(classifier))
        pipeline_args.append(strategy)

    pipeline = Pipeline(pipeline_args)

    # The internal cross-validation is used for hyperparameter optimization.
    print('Running hyperparameter optimization.')
    internal_cv = StratifiedKFold(n_splits=3)
    internal_evaluation = GridSearchCV(pipeline, hyperparameters, cv=internal_cv)
    internal_evaluation.fit(X_train, y_train)

    # The external cross-validation is used for model selection.
    external_cv = StratifiedKFold(n_splits=10)
    f1_weighted_scores_means = []

    # We perform the external cross-validation `num` times to 
    # mitigate wrong assumptions of the average scores of each model.
    num_times = 10

    for i in range(0, num_times):
        print('Running model selection cross-validation (' + str(i + 1) + ' out of ' + str(num_times) + ')')
        # Notice that the variable `internal_evaluation` represents 
        # the classification model with the best hyperparameters.
        f1_weighted_scores = cross_val_score(internal_evaluation, X_train, y_train, scoring='f1_weighted', cv=external_cv)
        f1_weighted_scores_means.append(f1_weighted_scores.mean())

    return internal_evaluation, f1_weighted_scores_means

def export_estimator_results(estimator_args, internal_evaluation, f1_weighted_scores_means, results_dir):
    filename = type(estimator_args['classifier']).__name__ + '_strategy_' + \
               estimator_args['strategy'] + '_oversample_' + str(estimator_args['oversample']).lower()

    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as results:
        # Estimator settings
        results.write('Estimator: ' + type(estimator_args['classifier']).__name__ + '\n')
        results.write('Strategy: ' + estimator_args['strategy'] + '\n')
        results.write('Oversample: ' + str(estimator_args['oversample']) + '\n\n')
        # Hyperparameter optimization report
        results.write('Internal Cross Validation (GridSearch)\n')
        results.write('Best parameters: ' + str(internal_evaluation.best_params_) + '\n')
        results.write('Best estimator: ' + str(internal_evaluation.best_estimator_) + '\n')
        # Model selection report 
        results.write('External Cross Validation (cross_val_score)\n')
        results.write('F1 weighted scores (averages): ' + str(f1_weighted_scores_means) + '\n')
