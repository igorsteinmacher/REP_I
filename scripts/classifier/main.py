#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
from matplotlib.pyplot import text

import pandas
from scipy.sparse import vstack
from data_preparation.import_data import import_data_for_classification, import_data_for_prediction
from model_selection.evaluate_estimators import evaluate_estimators_performance
from classification.train_model import train_classifier
from classification.explore_model import export_classification_report
from classification.explore_model import export_confusion_matrix 
from classification.explore_model import export_learning_curve 
from sklearn.svm import LinearSVC
import pickle

def find_best_estimator(X_train, y_train, results_dir):
    # Classifiers available:
    # 'svc' for Support Vector Classifier
    # 'mnb' for Multinomial Naive Bayes
    # 'knn' for K-nearest Neighbors
    # 'lr' for Logistic Regression
    # 'rf' for Random Forest
    # 'dmr' for Dummy Classifier (Random)
    # 'dmf' for Dummy Classifier (Always the most frequent)
    classifiers = ['dmf','svc', 'mnb', 'knn', 'lr', 'rf', 'dmr'] # 'svc', 'mnb', 'knn', 'lr', 'rf', 'dmr', 'dmf' 
    # Strategies available:
    # 'ovr' for OneVsRest
    # 'ovo' for OneVsOne
    strategies = ['ovr', 'ovo']
    # Oversampling:
    # True to apply SMOTE
    # False to not apply SMOTE
    oversample = [True, False]

    evaluate_estimators_performance(classifiers, strategies, oversample,
                                    X_train, y_train, results_dir)

def evaluate_final_estimator_on_unseen_data(X_train, y_train, X_test, y_test, results_dir):
    selected_classifier = LinearSVC(tol=0.001, C=1, max_iter=500)

    training_args = {
        'classifier': selected_classifier,
        'strategy': 'one_vs_rest',
        'oversample': False,
        'X_train': X_train,
        'y_train': y_train
    }

    model = train_classifier(**training_args)
    export_classification_report(model, X_test, y_test, results_dir)
    export_confusion_matrix(model, X_test, y_test)
    export_learning_curve(**training_args)

def train_final_estimator(X_train, y_train, X_test, y_test):
    selected_classifier = LinearSVC(tol=0.001, C=1, max_iter=500)

    X_train = vstack((X_train, X_test))
    y_train = pandas.concat([y_train, y_test])

    training_args = {
        'classifier': selected_classifier,
        'strategy': 'one_vs_rest',
        'oversample': False,
        'X_train': X_train,
        'y_train': y_train
    }

    model = train_classifier(**training_args)
    pickle.dump(model, open('final_estimator.sav', 'wb'))

def predict_using_final_estimator(data_dir):
    predict_spreadsheets_dir = os.path.join(data_dir, 'documentation', 'spreadsheets', 'prediction')
    X_train, _, X_test, _, train_text_column, test_text_column = import_data_for_prediction(predict_spreadsheets_dir, data_dir)
    X_predict = vstack((X_train, X_test))
    text_column = pandas.concat([train_text_column, test_text_column], ignore_index = True)

    model = pickle.load(open('final_estimator.sav', 'rb'))
    y_predict = model.predict(X_predict)

    for i in range(len(text_column)):
        print("\nText = %s\nClass Predicted = %s\n" % (text_column[i], y_predict[i]))
    
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

    X_train, y_train, X_test, y_test, _, _ = import_data_for_classification(spreadsheets_dir, data_dir)

    # find_best_estimator(X_train, y_train, X_test, y_test, results_dir)
    # evaluate_final_estimator_on_unseen_data(X_train, y_train, X_test, y_test, results_dir)
    train_final_estimator(X_train, y_train, X_test, y_test)
    predict_using_final_estimator(data_dir)