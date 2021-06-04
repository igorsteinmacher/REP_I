#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
from data_preparation.prepare_data import create_train_and_test_sets, import_set
from model_selection.evaluate_estimators import evaluate_estimators_performance
from model_selection.tune_hyperparameters import tune_hyperparameters
from results_report.report_estimators import export_estimators_performance

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def import_data_for_classification(spreadsheets_dir, data_dir):
    # Spreadsheets columns labels
    text_column = 'Paragraph'   
    classes_columns = ['No categories identified.',
                       'CF – Contribution flow',
                       'CT – Choose a task',
                       'TC – Talk to the community',
                       'BW – Build local workspace',
                       'DC – Deal with the code',
                       'SC – Submit the changes']

    # Label for a new column that merges classes_columns into a single one
    label_column = 'Label'

    # Filepaths where the train and test sets are saved
    train_filepath = os.path.join(data_dir, 'train.csv')
    test_filepath = os.path.join(data_dir, 'test.csv')

    # If files do not exist, create them
    if not os.path.exists(train_filepath) or not os.path.exists(test_filepath):
        create_train_and_test_sets(spreadsheets_dir, text_column, 
                                   classes_columns, label_column,
                                   data_dir)

    X_train, y_train = import_set(train_filepath, text_column, label_column)
    X_test, y_test = import_set(test_filepath, text_column, label_column)

    return X_train, y_train, X_test, y_test

def evaluate_estimators(strategies, classifiers, oversample, X_train, y_train, results_dir):
    strategies_available = {
        'ovr': 'one_vs_rest',
        'ovo': 'one_vs_one'
    }

    classifiers_available = {
        'rf': RandomForestClassifier(),
        'svc': LinearSVC(),
        'mnb': MultinomialNB(),
        'knn': KNeighborsClassifier(),
        'lr': LogisticRegression(),
    }

    hyperparameters_available = {
        'rf': {'clf__estimator__max_depth': [None, 50, 100],
               'clf__estimator__n_estimators': [50, 100, 150],
               'clf__estimator__min_samples_split': [1, 2, 3],
               'clf__estimator__min_samples_leaf': [1, 2, 3],
               'clf__estimator__max_features': ['auto', 'sqrt', 'log2'],
               'clf__estimator__max_leaf_nodes': [None, 50, 100]},
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

    selected_strategies = [strategies_available[strategy]
                           for strategy in strategies]

    selected_classifiers = [classifiers_available[classifier]
                            for classifier in classifiers]

    selected_hyperparameters = [hyperparameters_available[classifier]
                            for classifier in classifiers]

    estimator, performances = evaluate_estimators_performance(selected_classifiers,
                                                              selected_hyperparameters,
                                                              selected_strategies,
                                                              oversample, X_train, y_train)

    export_estimators_performance(performances, results_dir)

    return estimator

def tune_hyperparameters_of_best_estimator(estimator):
    best_hyperparameters = tune_hyperparameters(**estimator['args'])
    return best_hyperparameters.best_params_

if __name__ == '__main__':
    # Folders used during the classification process:
    # repository/scripts/classifier/
    classifier_dir = os.getcwd()
    # repository/scripts/
    scripts_dir = os.path.dirname(classifier_dir)
    # repository/
    repository_dir = os.path.dirname(scripts_dir) 
    # repository/data/
    data_dir = os.path.join(repository_dir, 'data')
    # repository/results/
    results_dir = os.path.join(repository_dir, 'results') 
    # repository/data/documentation/spreadsheets/valid
    spreadsheets_dir = os.path.join(data_dir, 'documentation', 'spreadsheets', 'valid')

    X_train, y_train, X_test, y_test = import_data_for_classification(spreadsheets_dir, data_dir)

    strategies = ['ovr'] # , 'ovo'
    classifiers = ['svc'] # , 'rf', 'mnb', 'knn', 'lr'
    oversample = [True] # , False

    best_estimator = evaluate_estimators(strategies, classifiers, oversample,
                                         X_train, y_train, results_dir)

    best_hyperparameters = tune_hyperparameters_of_best_estimator(best_estimator)
    print(best_hyperparameters)

