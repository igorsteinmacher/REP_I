#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
import scipy
from scipy.sparse import hstack
from import_data import import_dataframe
from prepare_data import text_preprocessing
from generate_features import create_statistic_features, create_heuristic_features
from train_model import train
from tune_parameters import hyperparameters_tuning
from deploy_model import report_performance, report_cross_validation, report_parameters, deploy_model
from cross_validation import k_fold_cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier # Google Text Classification Guidelines
from sklearn.linear_model import SGDClassifier # Scikit-learn Cheat Sheet

# TO-DO: Apply cross-validation methods.
# Plot confusion matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# Usefulness of features (most important features)
# TO-DO: Tune hyperparameters.

def multiclass_classification(preprocessing, classifiers, strategies, analysis_dir, 
                              results_dir, oversample=True, cross_validation=True, 
                              training=True, tune_parameters=True,
                              report=True, deploy=True):

    """Executes the multiclass classification process.

    The multiclass classification process is defined by a sequence of steps:
        Import Data ➜ Prepare Data ➜ Train Models ➜ Deploy Models and Performances

    Args:
        preprocessing: A list of preprocessing techniques to be applied.
        classifiers: A list of classifiers to train.
        strategies: A list of strategies to use during training.
        analysis_dir: A string representing the path to directory where the raw spreadsheets 
                    are available. In this context, raw spreadsheets are the data used
                    to train the classifier.
        results_dir: A string representing the path to directory where dataframes, 
                    models and performances should be saved.
        report: Boolean variable used to define if models peformances should be reported.
        deploy: Boolean variable used to define if models objects should be deployed.
    """
    ##########################
    #        SETUP           #
    ##########################
    text_column = 'Paragraph'
    label_column = 'Label'
    
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
        'sgd': SGDClassifier()
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
    dataframe = import_dataframe(analysis_dir, results_dir, classes)
    X = dataframe[text_column]
    y = dataframe[label_column]

    ##########################
    #     PREPARE DATA       #
    ##########################
    print("Applying preprocessing techniques on text column.")
    X = text_preprocessing(X, selected_preprocessing_techniques)
    
    print("Converting paragraphs into statistic features.")
    statistic_features = create_statistic_features(X)

    print("Converting paragraphs into heuristic features.")
    heuristic_features = create_heuristic_features(X)

    X = hstack([statistic_features, heuristic_features])

    ###################################
    # TRAIN, REPORT AND DEPLOY MODELS #
    ###################################
    for strategy in selected_training_strategies:
        print("Multiclass strategy: " + strategy )

        for oversample_status in oversample:
            print("Oversampling: " + str(oversample_status))

            for classifier in selected_classifiers:
                classifier_name = type(classifier).__name__
                print("Classifier: " + classifier_name)

                training_args = {
                    'classifier': classifier,
                    'strategy': strategy,
                    'oversample': oversample_status,
                    'X': X,
                    'y': y
                }

                deployment_args = {
                    'strategy': strategy,
                    'classifier_name': classifier_name,
                    'oversample_status': 'smote_' + str(oversample_status),
                    'results_dir': results_dir,
                }

                if cross_validation:
                    print("Running cross-validation")
                    scores = k_fold_cross_validation(**training_args)

                    if report:
                        print("Exporting cross-validation scores")
                        report_cross_validation(f1_scores=scores, **deployment_args)

                if tune_parameters:
                    print("Tuning hyper-parameters")
                    best_params = hyperparameters_tuning(**training_args)

                    if report:
                        print("Exporting best parameters set")
                        report_parameters(best_params=best_params, **deployment_args)

                if training:
                    print("Training classifier")
                    training_args['categories'] = classes
                    model, performance = train(**training_args)
                    del training_args['categories']

                    if report:
                        print("Exporting classification model performance")
                        report_performance(report=performance, **deployment_args)

                    if deploy:
                        print("Exporting classification model")
                        deploy_model(model=model, **deployment_args)

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    analysis_dir = os.path.join(root_dir, 'data', 'documentation', 'spreadsheets', 'done')
    results_dir = os.path.join(root_dir, 'results')

    training_args = {
        # Preprocessing Techniques
        # 'lc' to lowercase,
        # 'rp' to remove punctuations,
        # 'rs' to remove stopwords,
        # 'st' to use stemming (PorterStemmer NLTK),
        # 'lm' to use lemmatization (WordNetLemmatizer NLTK),
        'preprocessing': ['rs', 'st'], # 'rs', 'lc', 'rp', 'st'
        # Classification Algorithms:
        # 'rf' to use RandomForestClassifier,
        # 'svc' to use LinearSVC,
        # 'mnb' to use MultinomialNB,
        # 'knn' to use KNeighborsClassifier,
        # 'lr' to use LogisticRegression,
        # 'mlp' to use MLPClassifier,
        # 'sgd' to use SGDClassifier.
        'classifiers': ['svc'], # 'rf', 'mnb', 'knn', 'lr', 'mlp', 'sgd'
        # Training Strategies:
        # 'ovr' to use OneVsRest/OneVsAll,
        # 'ovo' to use OneVsOne.
        'strategies': ['ovr'], # 'ovo'
        # Oversampling: 
        # True to use oversampling, False otherwise.
        'oversample': [False], # , True
        'analysis_dir': analysis_dir,
        'results_dir': results_dir,
        'cross_validation': False,
        'tune_parameters': True,
        'training': False,
        'report': True,
        'deploy': False,
    }

    multiclass_classification(**training_args)