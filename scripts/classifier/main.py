#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
from data_preparation import prepare_data
from model_selection import evaluate_estimators

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

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
        prepare_data.create_sets_as_csv(spreadsheets_dir, text_column, 
                                        classes_columns, label_column,
                                        data_dir)

    X_train, y_train = prepare_data.import_set_as_dataframe(train_filepath, text_column, label_column)

    training_strategies_available = {
        'ovr': 'one-vs-rest',
        'ovo': 'one-vs-one'
    }

    classifiers_available = {
        'rf': RandomForestClassifier(),
        'svc': LinearSVC(),
        'mnb': MultinomialNB(),
        'knn': KNeighborsClassifier(),
        'lr': LogisticRegression(),
    }

    hyperparameters_available = {
        'rf': {'clf__estimator__max_depth': [None, 10, 100, 1000],
               'clf__estimator__n_estimators': [10, 100, 1000],
               'clf__estimator__min_samples_split': [1, 2, 5, 10],
               'clf__estimator__min_samples_leaf': [1, 2, 5, 10],
               'clf__estimator__max_features': ['auto', 'sqrt', 'log2'],
               'clf__estimator__max_leaf_nodes': [None, 10, 100, 1000]},
        'svc': {'clf__estimator__tol': [1e-4],
                'clf__estimator__C': [1],
                'clf__estimator__max_iter': [100]},
        'mnb': {'clf__estimator__alpha': [0.5, 1, 2, 5],
                'clf__estimator__fit_prior': [True, False]},
        'knn': {'clf__estimator__n_neighbors': [1, 2, 5, 10],
                'clf__estimator__metric': ['minkowski', 'euclidean', 'manhattan']},
        'lr': {'clf__estimator__tol': [1e-3, 1e-4, 1e-5],
               'clf__estimator__C': [1, 10, 100],
               'clf__estimator__max_iter': [100, 500, 1000]},
    }

    strategies = ['ovr', 'ovo']
    classifiers = ['svc', 'rf', 'mnb', 'knn', 'lr']
    oversample = [True, False]

    selected_training_strategies = [training_strategies_available[strategy]
                                    for strategy in strategies]

    selected_classifiers = [classifiers_available[classifier]
                            for classifier in classifiers]

    selected_hyperparameters = [hyperparameters_available[classifier]
                            for classifier in classifiers]

    estimators_performance = evaluate_estimators.estimators_performance(selected_classifiers,
                                                                        selected_hyperparameters,
                                                                        selected_training_strategies,
                                                                        oversample, X_train, y_train)
    print(estimators_performance)