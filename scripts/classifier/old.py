#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os

from import_data import import_dataframe
from select_model import evaluate_estimators_performance, hyperparameters_tuning
from train_model import train

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Plot ROC, AUC and MCC curves and values
# Update images 
#

def multiclass_training(preprocessing, classifiers, strategies, oversample, analysis_dir, results_dir):

    """Executes the multiclass training process.

    Our multiclass training process is defined by the sequence of steps:
    Import paragraphs
    ↳ Preprocess paragraphs 
     ↳ Extract heuristic and statistic features
      ↳ Split data in training and test sets
       ↳ Train a classification model using the training set
         ↳ Evaluate the performance of the model using the test set
    
    In the training stage, we:
    * Use nested ten-fold cross-validation to identify the classification
      algorithm that best fits our problem;
    * For the selected algorithm, use Gridsearch to tune the hyperparameters;
    * Train a final model using information from the two steps above.

    Args:
        preprocessing: A list of strings containing preprocessing techniques.
        classifiers: A list of strings containing classifiers to train.
        strategies: A list of strings containing multiclass strategies to use.
        analysis_dir: A string representing the path to the directory where the 
                    annotated spreadsheets are.
        results_dir: A string representing the path to the directory where
                    dataframes, models and performances should be saved.
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

    selected_preprocessing_techniques = [preprocessing_techniques_available[technique] 
                                         for technique in preprocessing]

    selected_training_strategies = [training_strategies_available[strategy]
                                    for strategy in strategies]

    selected_classifiers = [classifiers_available[classifier]
                            for classifier in classifiers]

    selected_hyperparameters = [hyperparameters_available[classifier]
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

    print("Splitting dataframe into training and test sets.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    ####################
    # TRAIN & VALIDATE #
    ####################
    print("Accessing estimators performance.")
    estimators_performance = evaluate_estimators_performance(selected_classifiers, 
                                                             selected_hyperparameters,
                                                             selected_training_strategies,
                                                             oversample,
                                                             X_train,
                                                             y_train)

    print("Identifying the best estimator.")
    highest_f1_mean = 0.0
    best_training_args = {}

    for estimator in estimators_performance:
        if estimators_performance[estimator]['f1_mean'] > highest_f1_mean:
            best_training_args = estimators_performance[estimator]['args']

    print("Tuning hyper-parameters for the best estimator.")
    best_training_args['hyperparameters'] = hyperparameters_tuning(**best_training_args)

    print("Training the classification model.")
    best_training_args['classes'] = classes
    best_training_args['X_test'] = X_test
    best_training_args['y_test'] = y_test
    train(**best_training_args)


if __name__ == '__main__':
    classifier_dir = os.getcwd() # repository_name/scripts/classifier/
    scripts_dir = os.path.dirname(classifier_dir) # repository_name/scripts/
    repository_dir = os.path.dirname(scripts_dir) # repository_name/
    analysis_dir = os.path.join(repository_dir, 'data', 'documentation', 'spreadsheets', 'valid')
    results_dir = os.path.join(repository_dir, 'results')

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
        'oversample': [True], # False
        'analysis_dir': analysis_dir,
        'results_dir': results_dir,
    }

    multiclass_training(**training_args)