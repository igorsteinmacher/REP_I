#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import numpy


def number_of_samples(dataframe):
    """Gets the number of samples in a dataframe.

    Args:
        dataframe: A pandas dataframe with samples for classification.

    Returns:
        An integer representing the total number of samples in the dataframe.
    """
    number_of_samples = dataframe.shape[0]
    return number_of_samples


def number_of_samples_per_class(dataframe, classes):
    """Counts the number of samples marked per class.

    Args:
        dataframe: A pandas dataframe with samples for classification.
        classes: A list of column names of the dataframe which represents
                the classes to be predicted.

    Returns:
        A dictionary that maps classes with the total number of samples marked.
    """
    samples_per_class = {}

    for label in classes:
        number_of_samples = dataframe[label].value_counts().get(1)
        samples_per_class[label] = number_of_samples

    return samples_per_class



def frequency_of_classes_per_sample(dataframe, classes):
    """Gets the frequency of classes for samples in a dataframe.

    Args:
        dataframe: A pandas dataframe with samples for classification.
        classes: A list of column names of the dataframe which represents
                the classes to be predicted.

    Returns:
        A dictionary that maps the number of classes with its frequency.
        For example:

        {5: 7,
        4: 3,
        2: 1}

        Means that seven samples were marked with five categories. Four samples
        with three categories , and two with one.
    """
    frequency = {}

    for index, row in dataframe.iterrows():
        number_of_classes = 0
        for label in classes:
            if row[label] == 1:
                number_of_classes += 1
        if number_of_classes in frequency.keys():
            frequency[number_of_classes] += 1
        else:
            frequency[number_of_classes] = 1

    return frequency


def median_of_words_per_sample(dataframe):
    """Gets the median number of words per sample.

    Args:
        dataframe: A pandas dataframe with samples for classification.

    Returns:
        A float representing the median number of words per sample.
    """
    words_per_sample = [len(row['paragraph'].split())
                        for index, row in dataframe.iterrows()]
    return numpy.median(words_per_sample)
