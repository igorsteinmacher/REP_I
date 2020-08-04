import pandas
import matplotlib.pyplot as plt
import numpy as np
import os
plt.style.use('seaborn')

class ExperimentAnalysis:
    def __init__(self, root):
        self.dataframes = {'README': [], 'CONTRIBUTING': []}
        self.categories = ['CF – Contribution flow',
                           'CT – Choose a task',
                           'FM – Find a mentor',
                           'TC – Talk to the community',
                           'BW – Build local workspace',
                           'DC – Deal with the code',
                           'SC – Submit the changes']

        for folder in os.listdir(root):
            if os.path.isdir(root + folder):
                for filename in os.listdir(root + folder):
                    file_path =  root + folder + '/' + filename
                    readme = pandas.read_excel(file_path, sheet_name='README')
                    contributing = pandas.read_excel(file_path, sheet_name='CONTRIBUTING')
                    self.dataframes['README'].append(readme)
                    self.dataframes['CONTRIBUTING'].append(contributing)

    def validate_dataframes(self, dataframes):
        for dataframe in dataframes:
            if dataframe.empty:
                raise Exception('The dataframe is empty. File: ' + filename + '.')
            elif list(dataframe.columns.values) != self.categories:
                raise Exception('Unknown column value. File: ' + filename + '.')
    
    def categories_frequency(self):
        frequency = pandas.DataFrame(0, index=self.dataframes.keys(), columns=self.categories)

        for filetype in self.dataframes.keys():
            for dataframe in self.dataframes[filetype]:
                for column in dataframe.columns:
                    frequency.at[filetype, column] = frequency.loc[filetype, column] + dataframe[column].count()

        fig, ax = plt.subplots()
        labels = np.arange(frequency.shape[1])
        readme_bar = ax.bar(labels, frequency.loc['README'], width=0.35)
        contributing_bar = ax.bar(labels + 0.35, frequency.loc['CONTRIBUTING'], width=0.35)
        ax.set_ylabel('Occurrencess')
        ax.set_xticks(labels + 0.35)
        ax.set_xticklabels([label[0:2] for label in frequency.columns])
        ax.legend((readme_bar[0], contributing_bar[0]), ('README', 'CONTRIBUTING'))
        plt.show()

experiment = ExperimentAnalysis('../analysis/usp/')
experiment.categories_frequency()