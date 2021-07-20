#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
import joblib

def deploy_model(model, strategy, classifier_name, results_dir):
    """Deploys a classification model as a joblib for prediction purposes

    Args:
        model: A scikit-learn multiclass classification model 
        strategy: A string representing the strategy used in the multiclass classification
        classifier_name: A string representing the classification algorithm name
        results_dir: A string representing the path to directory where dataframes, 
                    models and performances should be saved.
    """
    model_filename = classifier_name + '.joblib'
    output_dir = os.path.join(results_dir, 'models', strategy)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
