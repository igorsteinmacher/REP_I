#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
from data_preparation.import_data import import_data_for_classification
from model_selection.evaluate_estimators import evaluate_estimators_performance
from classification.train_model import train_classifier
from classification.explore_model import export_classification_report
from classification.explore_model import export_confusion_matrix 
from classification.explore_model import export_learning_curve 
from sklearn.svm import LinearSVC

# Pega a tabela com os resultados de cada classificador
# Divide os algoritmos entre com oversampling e sem oversampling
# Pra cada algoritmo (melhor configuracao ja definida), roda ele no conjunto de teste usando predict_proba
# Cria uma tabela onde as colunas sao os algoritmos (OvA, OvO) e as linhas sao os samples/paragrafos
# Cada linha contem o valor da probabilidade da classe que seria correta (nao da classe predita/majoritaria) pro respectivo algoritmo na coluna

#               RandomForest (OvO) RandomForest (OvA) LinearSVC (OvO) LinearSVC(OvA)
# Paragrafo A
# Paragrafo B
# Paragrafo C
# Paragrafo D
# Paragrafo E

def estimator_selection(X_train, y_train, X_test, y_test, results_dir):
    # Classifiers available:
    # 'svc' for Support Vector Classifier
    # 'mnb' for Multinomial Naive Bayes
    # 'knn' for K-nearest Neighbors
    # 'lr' for Logistic Regression
    # 'rf' for Random Forest
    # 'dmr' for Dummy Classifier (Random)
    # 'dmf' for Dummy Classifier (Always the most frequent)
    classifiers = [] # 'svc', 'mnb', 'knn', 'lr', 'rf', 'dmr', 'dmf' 
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

def final_training(X_train, y_train, X_test, y_test, results_dir):
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

    estimator_selection(X_train, y_train, X_test, y_test, results_dir)
    # final_training(X_train, y_train, X_test, y_test, results_dir)