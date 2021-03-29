#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
from import_data import import_dataframe
from prepare_data import shuffle_and_split, text_preprocessing, vectorize_paragraphs
from train_model import train
from deploy_model import report_performance, deploy_model

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier # Google Text Classification Guidelines
from sklearn.linear_model import SGDClassifier # Scikit-learn Cheat Sheet

# TO-DO: Extract heuristic features.
# TO-DO: Perform a feature selection for X_train and X_test.
# TO-DO: Tune hyperparameters.
# TO-DO: Apply cross-validation methods.

def multiclass_classification(preprocessing, classifiers, strategies, analysis_dir, results_dir, report=True, deploy=True):
    """Executes the multiclass classification process.

    The multiclass classification process is defined by a sequence of steps:
        Import Data ➜ Prepare Data ➜ Train Models ➜ Deploy Models and Performances

    Args:
        preprocessing: A list of preprocessing techniques to be applied.
        classifiers: A list of classifiers to train.
        strategies: A list of strategies to use during training.
        analysis_dir: A string representing the path to directory where raw spreadsheets 
                    are available. In this context, raw spreadsheets are the data used
                    to train the classifier.
        results_dir: A string representing the path to directory where dataframes, 
                    models and performances should be saved.
        report: Boolean value used to define if models peformances should be reported.
        deploy: Boolean value used to define if models trained objects should be deployed.
    """
    ##########################
    #        SETUP           #
    ##########################
    text_column = 'Paragraph'
    label_column = 'label'
    
    classes = ['No categories identified.',
               'CF – Contribution flow',
               'CT – Choose a task',
               'TC – Talk to the community',
               'BW – Build local workspace',
               'DC – Deal with the code',
               'SC – Submit the changes']

    preprocessing_techniques_available = {
        'lc': 'lowercase',
        'rp': 'remove-punctuations',
        'rs': 'remove-stopwords',
        'st': 'stemming',
        'lm': 'lemmatization'
    }

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
        'mlp': MLPClassifier(),
        'sgd': SGDClassifier(),
    }

    selected_preprocessing_techniques = [preprocessing_techniques_available[technique] 
                                         for technique in preprocessing]

    selected_training_strategies = [training_strategies_available[strategy]
                                    for strategy in strategies]

    selected_classifiers = [classifiers_available[classifier]
                            for classifier in classifiers]

    #########################
    #      IMPORT DATA      #
    #########################
    print("Importing data for classification.")
    dataframe = import_dataframe(analysis_dir, results_dir)

    ##########################
    #     PREPARE DATA       #
    ##########################
    print("Applying preprocessing techniques on text column.")
    dataframe = text_preprocessing(dataframe, text_column, selected_preprocessing_techniques)

    print("Shuffling and splitting dataframe into training and test sets.")
    X_train, X_test, y_train, y_test = shuffle_and_split(
        dataframe, [text_column], label_column)
        
    print("Converting paragraphs to statistical features.")
    X_train, X_test = vectorize_paragraphs(
        X_train[text_column].tolist(), X_test[text_column].tolist())

    ###################################
    # TRAIN, REPORT AND DEPLOY MODELS #
    ###################################
    for strategy in selected_training_strategies:
        print("Executing " + strategy + " classification strategy.")

        for classifier in selected_classifiers:
            classifier_name = type(classifier).__name__
            print("Training " + classifier_name + " using " + strategy + " strategy.")

            training_args = {
                'categories': classes,
                'classifier': classifier,
                'strategy': strategy,
                'oversample': True,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
            }

            deployment_args = {
                'strategy': strategy,
                'classifier_name': classifier_name,
                'results_dir': results_dir,
            }

            model, performance = train(**training_args)

            if report:
                print("Exporting results of " + classifier_name + " using " + strategy + " strategy.")
                report_performance(report=performance, **deployment_args)

            if deploy:
                print("Deploying model of " + classifier_name + " using " + strategy + " strategy.")
                deploy_model(model=model, **deployment_args)

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    analysis_dir = os.path.join(root_dir, 'data', 'documentation', 'spreadsheets', 'done')
    results_dir = os.path.join(root_dir, 'results')

    training_args = {
        'classifiers': ['svc', 'mnb', 'knn', 'lr'], # 'rf', 'mlp'
        'strategies': ['ovo', 'ovr'],
        'analysis_dir': analysis_dir,
        'results_dir': results_dir,
        'report': True,
        'deploy': False,
    }

    multiclass_classification(**training_args)