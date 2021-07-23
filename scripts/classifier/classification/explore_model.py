
import os
import json
import matplotlib.pyplot as plot
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve

import numpy as np
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from imblearn.over_sampling import SMOTE

def export_classification_report(model, X_test, y_test, results_dir):
    classes = ['No categories identified.',
               'CT – Choose a task',
               'TC – Talk to the community',
               'BW – Build local workspace',
               'DC – Deal with the code',
               'SC – Submit the changes']

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)

    report_filepath = os.path.join(results_dir, 'classification_report.json')

    with open(report_filepath, 'w') as report_file:
        json.dump(report, report_file, indent=4)

def export_confusion_matrix(model, X_test, y_test):
    classes = ['No categories identified.',
               'CT – Choose a task',
               'TC – Talk to the community',
               'BW – Build local workspace',
               'DC – Deal with the code',
               'SC – Submit the changes']

    labels = [category[:2] for category in classes]
    plot_confusion_matrix(model, X_test, y_test, display_labels=labels, cmap=plot.cm.Blues)
    plot.show()


def export_learning_curve(classifier, strategy, oversample, X_train, y_train):
    plt.figure()

    if strategy == 'one_vs_rest':
        model = OneVsRestClassifier(classifier)
    if strategy == 'one_vs_one':
        model = OneVsOneClassifier(classifier)

    if oversample:
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, train_sizes=np.linspace(0.1, 1, 10, 20), cv=10)
    plt.plot(train_sizes, -test_scores.mean(1), 'o-', color="r",
            label="LinearSVC")
    plt.xlabel("Train size")
    plt.ylabel("Mean Squared Error")
    plt.title('Learning curves')
    plt.legend(loc="best")
    plt.show()