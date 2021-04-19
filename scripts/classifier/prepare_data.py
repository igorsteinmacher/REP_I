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
    """Applies text processing techniques to a dataframe column of strings.

    Before converting paragraphs into features, a good starting point may be to apply
    pre-processing techniques that will remove unwanted information. This method contains
    a series of submethods that represent common preprocessing techniques used in
    machine learning.
    
    X: A dataframe column of strings with text for classification.
    techniques: A dictionary with keys representing techniques to be applied in the 
        text processing process.
    
    Returns:
        The dataframe column of strings updated with the values formated by the preprocessing
         techniques desired.
    """
    def lowercase(paragraph):
        # Transform uppercase characters into lowercase
        return paragraph.lower()

    def remove_punctuations(paragraph):
        # Remove all the punctuations of the text, including: !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
        return paragraph.translate(str.maketrans('', '', string.punctuation))

    def remove_stopwords(paragraph):
        # Remove all the stopwords of the paragraph, such as: "the, for, but, nor"
        stop_words = set(stopwords.words('english'))
        return " ".join([word for word in paragraph.split() if word not in stop_words])

    def stemming(paragraph):
        # Read about stemming at:
        # nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
        stemmer = PorterStemmer()
        return " ".join([stemmer.stem(word) for word in paragraph.split()])

    def lemmatization(paragraph):
        # Read about lemmatization at:
        # nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
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
    X: A dataframe column of strings with text features for classification.
    y: A dataframe column of strings representing labels for classification.

    Notice that the text provided by X matches with the labels provided by y.

    Returns:
        A column of features for training (X_train), a column of features
        for test (X_test), a column of labels for training (y_train)
        and a column of labels for test (y_test).
    """

    # Google recommends 80% of the samples for training, and 20% for validation.
    # developers.google.com/machine-learning/guides/text-classification/step-3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True)

    return X_train, X_test, y_train, y_test
