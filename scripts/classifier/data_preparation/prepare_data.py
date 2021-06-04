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
                               label_column, export_dir):

    dataframe = transform_spreadsheets_in_dataframe(spreadsheets_dir,
                                                    text_column,
                                                    classes_columns,
                                                    label_column)

    X = dataframe[text_column]
    y = dataframe[label_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    train_data = pandas.concat([X_train, y_train], axis=1)
    test_data = pandas.concat([X_test, y_test], axis=1)

    train_filepath = os.path.join(export_dir, 'train.csv')
    test_filepath = os.path.join(export_dir, 'test.csv')

    train_data.to_csv(train_filepath, index=False, encoding='utf-8-sig')
    test_data.to_csv(test_filepath, index=False, encoding='utf-8-sig')

def import_set(csv_filepath, text_column, label_column):
    data = pandas.read_csv(csv_filepath)
    X, y = data[text_column], data[label_column]

    print("Applying preprocessing techniques on paragraphs column.")
    preprocessing_techniques = ['remove-stopwords', 'lemmatization']
    X = text_preprocessing(X, preprocessing_techniques)
    
    print("Converting paragraphs into statistic features.")
    statistic_features = create_statistic_features(X)

    print("Converting paragraphs into heuristic features.")
    heuristic_features = create_heuristic_features(X)

    X = hstack([statistic_features, heuristic_features])

    return X, y
