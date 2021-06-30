#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os

from sklearn.svm import LinearSVC
from data_preparation.import_data import import_data_for_classification
from model_selection.evaluate_estimators import evaluate_estimators_performance
from classification.train_model import train_classifier
from classification.explore_model import export_classification_report
from classification.explore_model import export_confusion_matrix
from classification.explore_model import export_roc_curve

def run_classification_method(estimator_selection, classification):
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

    if estimator_selection:
        # Classifiers available:
        # 'svc': Support Vector Classifier
        # 'mnb': Multinomial Naive Bayes
        # 'knn': K-nearest Neighbors
        # 'lr': Logistic Regression
        # 'rf': Random Forest
        classifiers = ['svc', 'mnb', 'knn', 'lr', 'rf'] 
        # Strategies available:
        # 'ovr': OneVsRest
        # 'ovo': OneVsOne
        strategies = ['ovr', 'ovo']
        # Oversampling:
        # True to run with SMOTE
        # False to run without SMOTE
        oversample = [True, False]

        evaluate_estimators_performance(classifiers, strategies, oversample,
                                        X_train, y_train, results_dir)

    if classification:
        training_args = {
            'classifier': LinearSVC(tol=0.001, C=1, max_iter=500),
            'strategy': 'one_vs_rest',
            'oversample': False,
            'X_train': X_train,
            'y_train': y_train
        }

        model = train_classifier(**training_args)
        export_classification_report(model, X_test, y_test, results_dir)
        # export_confusion_matrix(model, X_test, y_test)
        # export_roc_curve(model, X_test, y_test)

if __name__ == '__main__':
    run_classification_method(True, True)