
import os
import json
import matplotlib.pyplot as plot
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

def report_classification(model, X_test, y_test, results_dir):
    classes = ['No categories identified.',
               'CF – Contribution flow',
               'CT – Choose a task',
               'TC – Talk to the community',
               'BW – Build local workspace',
               'DC – Deal with the code',
               'SC – Submit the changes']

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)

    report_filepath = os.path.join(results_dir, 'classification_report.json')

    with open(report_filepath, 'w') as report_file:
        json.dump(report, report_file)

def plot_confusion_matrix(model, X_test, y_test):
    classes = ['No categories identified.',
               'CF – Contribution flow',
               'CT – Choose a task',
               'TC – Talk to the community',
               'BW – Build local workspace',
               'DC – Deal with the code',
               'SC – Submit the changes']

    labels = [category[:2] for category in classes]
    plot_confusion_matrix(model, X_test, y_test, display_labels=labels, cmap=plot.cm.Blues)
    plot.show()

