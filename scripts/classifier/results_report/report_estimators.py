import os
from csv import DictWriter

def export_estimators_performance(estimators_performance, export_dir):
    csv_filepath = os.path.join(export_dir, 'estimators_performance.csv')

    with open(csv_filepath, 'w') as csv_file:
        fieldnames = ['estimator', 'f1_mean']
        csv_writer = DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        for estimator in estimators_performance:
            row = {'estimator': estimator.lower(),
                   'f1_mean': estimators_performance[estimator]['f1_mean']}

            csv_writer.writerow(row)