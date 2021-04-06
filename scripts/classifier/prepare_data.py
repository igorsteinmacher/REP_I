#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

# Preprocessing
import spacy
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import string
# Preparation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def text_preprocessing(dataframe, text_column, techniques = {}):
    if 'lowercase' in techniques:
        def lowercase(paragraph):
            return paragraph.lower()

        dataframe[text_column] = dataframe[text_column].apply(lambda paragraph: lowercase(paragraph))

    if 'remove-punctuations' in techniques:
        def remove_punctuations(paragraph):
            return paragraph.translate(str.maketrans('', '', string.punctuation))
        
        dataframe[text_column] = dataframe[text_column].apply(lambda paragraph: remove_punctuations(paragraph))

    if 'remove-stopwords' in techniques:
        def remove_stopwords(paragraph):
            stop_words = set(stopwords.words('english'))
            return " ".join([word for word in paragraph.split() if word not in stop_words])

        dataframe[text_column] = dataframe[text_column].apply(lambda paragraph: remove_stopwords(paragraph))
    
    if 'stemming' in techniques:
        def stemming(paragraph):
            stemmer = PorterStemmer()
            return " ".join([stemmer.stem(word) for word in paragraph.split()])

        dataframe[text_column] = dataframe[text_column].apply(lambda paragraph: stemming(paragraph))

    if 'lemmatization' in techniques:
        if 'stemming' in techniques:
            print('Warning: You are using both lemmatization and stemming.')

        def lemmatization(paragraph):
            lemmatizer = WordNetLemmatizer()
            return " ".join([lemmatizer.lemmatize(word) for word in paragraph.split()])

        dataframe[text_column] = dataframe[text_column].apply(lambda paragraph: lemmatization(paragraph))

    if 'spacy-lemmatization' in techniques:
        if 'stemming' in techniques:
            print('Warning: You are using both lemmatization and stemming.')

        if 'lemmatization' in techniques:
            print('Warning: You are already using lemmatization.')

        def spacy_lemmatization(paragraph):
            nlp = spacy.load("en_core_web_sm")
            lemmatizer = nlp.get_pipe("lemmatizer")
            doc = nlp(paragraph)
            return " ".join([token.lemma_ for token in doc])

        dataframe[text_column] = dataframe[text_column].apply(lambda paragraph: spacy_lemmatization(paragraph))

    return dataframe

def shuffle_and_split(dataframe, features, labels, smote_is_active=True):
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
    X = dataframe[features]
    y = dataframe[labels]

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