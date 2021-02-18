#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
from gather_data import check_if_dataframe_copy_exists
from prepare_data import shuffle_and_split, vectorize_paragraphs
from train_model import train
from deploy_model import report_performance, deploy_model

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier # Google Text Class. Guidelines
from sklearn.linear_model import SGDClassifier # Scikit-learn Cheat Sheet

# TO-DO: Extract heuristic features
# TO-DO: Perform a feature selection for X_train and X_test
# TO-DO: Tune hyperparameters
# TO-DO: Apply cross-validation methods

def classify(classifiers, strategies, report, deploy, results_dir, analysis_dir):
    """Executes the classification process.

    The classification process is defined by a sequence of steps:
        Start ➜ Gather Data ➜ Prepare Data ➜ Train Models ➜ Deploy Models and Performances ➜ End

    Args:
        classifiers: A list of classifiers to train.
        strategies: A list of strategies to use during training.
        report: Boolean value used to define if models peformances will be reported.
        deploy: Boolean value used to define if models trained objects will be deployed.
        results_dir: A string path to directory where dataframes, models and performances
                    will be saved.
        analysis_dir: A string path to directory where raw spreadsheets are available.
    """
    ##########################
    #        START           #
    ##########################
    text_column = 'Paragraph'
    
    classes = ['CF – Contribution flow',
               'CT – Choose a task',
               'TC – Talk to the community',
               'BW – Build local workspace',
               'DC – Deal with the code',
               'SC – Submit the changes']

    classifiers_available = {
        'rf': RandomForestClassifier(),
        'svc': LinearSVC(),
        'mnb': MultinomialNB(),
        'knn': KNeighborsClassifier(),
        'lr': LogisticRegression(),
        'mlp': MLPClassifier(),
        'sgd': SGDClassifier(),
    }

    strategies_available = {
        'cc': 'classifier-chains',
        'lp': 'label-powerset',
        'ovr': 'one-vs-the-rest'
    }
    selected_classifiers = [classifiers_available[classifier]
                            for classifier in classifiers]
    selected_strategies = [strategies_available[strategy]
                           for strategy in strategies]

    ##########################
    #      GATHER DATA       #
    ##########################
    print("Gathering dataframe for classification.")
    dataframe = check_if_dataframe_copy_exists(results_dir, analysis_dir)

    ##########################
    #     PREPARE DATA       #
    ##########################
    print("Shuffling and splitting dataframe into training and test sets.")
    X_train, X_test, y_train, y_test = shuffle_and_split(
        dataframe, [text_column], classes)
    print("Converting paragraphs to statistical features.")
    X_train, X_test = vectorize_paragraphs(
        X_train[text_column].tolist(), X_test[text_column].tolist())

    ###################################
    # TRAIN, REPORT AND DEPLOY MODELS #
    ###################################
    for classifier in selected_classifiers:
        classifier_name = type(classifier).__name__
        training_args = {
            'classifier': classifier,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
        }

        for strategy in selected_strategies:
            print("Executing " + strategy + " classification.")
            print("Classifier: " + classifier_name)
            deployment_args = {
                'strategy': strategy,
                'classifier_name': classifier_name,
                'results_dir': results_dir,
            }

            model, performance = train(strategy=strategy, **training_args)
            if report:
                report_performance(report=performance, **deployment_args)
            if deploy:
                deploy_model(model=model, **deployment_args)


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    analysis_dir = os.path.join(root_dir, 'data', 'documentation', 'spreadsheets', 'done')
    results_dir = os.path.join(root_dir, 'results')

    kwargs = {
        'classifiers': ['sgd', 'rf' , 'svc', 'mnb', 'knn', 'lr', 'mlp'],
        'strategies': ['ovr'], # , 'cc', 'lp'
        'report': True,
        'deploy': True,
        'results_dir': results_dir,
        'analysis_dir': analysis_dir
    }
    classify(**kwargs)