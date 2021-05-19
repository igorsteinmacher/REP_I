#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
import json
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

    joblib.dump(model, os.path.join(output_dir, model_filename))

def report_cross_validation(f1_scores, strategy, classifier_name, oversample_status, results_dir):
    output_dir = os.path.join(results_dir, 'cross-validation', strategy, oversample_status)
    report_filename = classifier_name + '.txt'

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, report_filename), 'w') as file:
        file.write(str(f1_scores))

def report_performance(report, strategy, classifier_name, oversample_status, results_dir):
    """Reports the performance of a classification model as a JSON for analysis

    Args:
        model: A JSON performance report from a scikit-learn multiclass classification model
        strategy: A string representing the strategy used in the multiclass classification
        classifier_name: A string representing the classification algorithm name
        results_dir: A string representing the path to directory where dataframes, 
                    models and performances should be saved.   
    """
    report['strategy'] = strategy
    report['classifier_name'] = classifier_name
    report_filename = classifier_name + '.report.json'

    output_dir = os.path.join(results_dir, 'models', strategy, oversample_status)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, report_filename), 'w') as file:
        file.write(json.dumps(report, indent=4, sort_keys=True))
