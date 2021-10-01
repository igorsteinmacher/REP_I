#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import pandas
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split

from .generate_features import create_statistic_features, create_heuristic_features
from .transform_data import transform_spreadsheets_in_dataframe
from .preprocess_text import text_preprocessing
from .select_features import select_features

def create_train_and_test_sets(spreadsheets_dir, text_column, classes_columns,
                               train_filepath, test_filepath, label_column):
    """Creates the train and test sets based on the spreadsheets from the 
        qualitative analysis.

    Args:
        spreadsheets_dir: A string representing the path to the spreadsheets folder
        text_column: A string representing what is the column containing the paragraphs
            in each spreadsheet
        classes_columns: A list of strings representing what are the columns representing
            classes in each spreadsheet
        train_filepath: A string representing the filepath where the train set should be
            saved as a .csv file
        test_filepath: A string representing the filepath where the test set should be
            saved as a .csv file
        label_column: A string representing the label that will be given to a new column that
            will be used to represent the label of each paragraph.
    """

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

def import_sets(train_filepath, test_filepath, text_column, label_column, is_predict = False):
    """Imports train and test sets and applies the text preprocessing techniques when necessary
    
    Args:
        text_column: A string representing what is the column containing the paragraphs
        in each spreadsheet
        train_filepath: A string representing the filepath where the train set should be
            saved as a .csv file
        test_filepath: A string representing the filepath where the test set should be
            saved as a .csv file
        label_column: A string representing the label that will be given to a new column that
            will be used to represent the label of each paragraph.
    """
    print("Importing training and test sets.")
    train_data = pandas.read_csv(train_filepath)
    train_text_column = train_data[text_column]
    X_train, y_train = train_data[text_column], train_data[label_column]

    test_data = pandas.read_csv(test_filepath)
    test_text_column = test_data[text_column]
    X_test, y_test = test_data[text_column], test_data[label_column]

    print("Applying preprocessing techniques on paragraphs column.")
    preprocessing_techniques = ['remove-stopwords', 'remove-punctuations', 'lemmatization']
    X_train = text_preprocessing(X_train, preprocessing_techniques)
    X_test = text_preprocessing(X_test, preprocessing_techniques)
    
    print("Converting paragraphs into statistic features.")
    train_statistic_features, test_statistic_features = create_statistic_features(X_train, X_test, is_predict)

    print("Converting paragraphs into heuristic features.")
    train_heuristic_features, test_heuristic_features = create_heuristic_features(X_train, X_test)

    X_train = hstack([train_statistic_features, train_heuristic_features])
    X_test = hstack([test_statistic_features, test_heuristic_features])

    print("Selecting features with SelectPercentile (chi2).")
    X_train, X_test = select_features(X_train, y_train, X_test, is_predict)

    return X_train, y_train, X_test, y_test, train_text_column, test_text_column
