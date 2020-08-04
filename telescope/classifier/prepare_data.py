#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np


def shuffle_and_split(dataframe, features, classes):
    """Shuffles the dataframe and splits it into training and testing sets.

    Args:
        dataframe: A dataframe with samples for classification.
        text_column: A string representing the text column in the dataframe.
        classes: A list of strings representing the columns of labels in the
                dataframe.

    Returns:
        A dataframe of features for training (X_train), a dataframe of features
        for test (X_test), a one-column dataframe of labels for training (y_train)
        and a one-column dataframe of labels for test (y_test).

    Note:
        In our case, we are preliminary considering as features only the text
        column. This column will be transformed into a set of TF-IDF features
        after the vectorization.
    """
    X = dataframe[features]
    y = dataframe[classes]

    # Google recommends 80% of the samples for training, and 20% for validation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True)

    return X_train, X_test, y_train, y_test


def vectorize_paragraphs(X_train, X_test):
    """Converts raw training and test paragraphs to TF-IDF features.

    Args:
        X_train: A list of paragraphs for training.
        X_test: A list of paragraphs for test.

    Returns:
        A matrix of TF-IDF features for training and a matrix of TF-IDF features
        for test.
    """
    vect_args = {
        'ngram_range': (1, 2),  # Google recomends: 1-gram + 2-grams
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'stop_words': 'english',
        'analyzer': 'word',
    }
    vectorizer = TfidfVectorizer(**vect_args)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    return X_train, X_test
