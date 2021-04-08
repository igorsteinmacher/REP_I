#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

# Preprocessing
from spacy.lang.en import English
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import string
# Preparation
from sklearn.model_selection import train_test_split

def text_preprocessing(X, techniques = {}):
    def lowercase(paragraph):
        return paragraph.lower()

    def remove_punctuations(paragraph):
        return paragraph.translate(str.maketrans('', '', string.punctuation))

    def remove_stopwords(paragraph):
        stop_words = set(stopwords.words('english'))
        return " ".join([word for word in paragraph.split() if word not in stop_words])

    def stemming(paragraph):
        stemmer = PorterStemmer()
        return " ".join([stemmer.stem(word) for word in paragraph.split()])

    def lemmatization(paragraph):
        lemmatizer = WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(word) for word in paragraph.split()])

    if 'lowercase' in techniques:
        X = X.apply(lambda paragraph: lowercase(paragraph))

    if 'remove-punctuations' in techniques:        
        X = X.apply(lambda paragraph: remove_punctuations(paragraph))

    if 'remove-stopwords' in techniques:
        X = X.apply(lambda paragraph: remove_stopwords(paragraph))
    
    if 'stemming' in techniques:
        X = X.apply(lambda paragraph: stemming(paragraph))

    if 'lemmatization' in techniques:
        X = X.apply(lambda paragraph: lemmatization(paragraph))

    return X

def shuffle_and_split(X, y):
    """Shuffles the dataframe and splits it into training and testing sets.

    Args:
        dataframe: A dataframe with samples for classification.
        features: A string representing the text column in the dataframe.
        labels: A string representing the column of labels in the
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

    # Google recommends 80% of the samples for training, and 20% for validation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True)

    return X_train, X_test, y_train, y_test
