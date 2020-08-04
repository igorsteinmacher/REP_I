#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
import json
import joblib

def deploy_model(model, strategy, classifier_name, results_dir):
    model_filename = classifier_name + '.joblib'
    output_dir = os.path.join(results_dir, 'models', strategy)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    joblib.dump(model, os.path.join(output_dir, model_filename))

def report_performance(report, strategy, classifier_name, results_dir):
    report['strategy'] = strategy
    report['classifier_name'] = classifier_name
    report_filename = classifier_name + '.report.json'

    output_dir = os.path.join(results_dir, 'models', strategy)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, report_filename), 'w') as file:
        file.write(json.dumps(report, indent=4, sort_keys=True))
