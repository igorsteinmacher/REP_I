#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
import pandas
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from .preprocess_text import text_preprocessing
from .generate_features import create_statistic_features, create_heuristic_features
from .transform_data import transform_spreadsheets_in_dataframe

def create_train_and_test_sets(spreadsheets_dir, text_column, classes_columns,
                               train_filepath, test_filepath, label_column, data_dir):

    dataframe = transform_spreadsheets_in_dataframe(spreadsheets_dir,
                                                    text_column,
                                                    classes_columns,
                                                    label_column)

    X = dataframe[text_column]
    y = dataframe[label_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    train_data = pandas.concat([X_train, y_train], axis=1)
    test_data = pandas.concat([X_test, y_test], axis=1)

    train_data.to_csv(train_filepath, index=False, encoding='utf-8-sig')
    test_data.to_csv(test_filepath, index=False, encoding='utf-8-sig')

def import_sets(train_filepath, test_filepath, text_column, label_column):
    print("Importing training and test sets.")
    train_data = pandas.read_csv(train_filepath)
    X_train, y_train = train_data[text_column], train_data[label_column]

    test_data = pandas.read_csv(test_filepath)
    X_test, y_test = test_data[text_column], test_data[label_column]

    print("Applying preprocessing techniques on paragraphs column.")
    preprocessing_techniques = ['remove-stopwords', 'lemmatization']
    X_train = text_preprocessing(X_train, preprocessing_techniques)
    X_test = text_preprocessing(X_test, preprocessing_techniques)
    
    print("Converting paragraphs into statistic features.")
    train_statistic_features, test_statistic_features = create_statistic_features(X_train, X_test)

    print("Converting paragraphs into heuristic features.")
    train_heuristic_features, test_heuristic_features = create_heuristic_features(X_train, X_test)

    X_train = hstack([train_statistic_features, train_heuristic_features])
    X_test = hstack([test_statistic_features, test_heuristic_features])

    return X_train, y_train, X_test, y_test
