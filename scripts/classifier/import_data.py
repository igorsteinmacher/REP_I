#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
import pandas


def import_dataframe(analysis_dir, results_dir, classes):
    """Imports the data that will be used for training as a pandas dataframe.
   
    First of all, it is important to understand that the data used in our study
    is primarily represented by a folder of spreadsheets which were manually analyzed
    by software engineering researchers. These spreadsheets follow a same standard:
    each contains a single worksheet named as `contributing` with seven pre-defined columns
    and a different number of rows.

    To avoid parsing all the spreadsheets in every execution (which takes time and memory),
    we parse the spreadsheets at the first execution and save a .xlsx copy of the dataframe 
    inside the `results_dir` directory.

    If this copy already exists, the .xlsx file is directly converted to a pandas dataframe. 
    Otherwise, the parsing methods are executed on the spreadsheets available inside the
    `analysis_dir` and a new copy of the dataframe is prepared.

    Args:
        analysis_dir: A string representing a path to the directory containing the 
                      spreadsheets used during the qualitative analysis.
        results_dir: A string representing a path to the directory where the 
                     dataframe may be saved.
        classes: A list of strings containing the columns that should be extracted from each spreadsheet
            as classes of the classifier.

    Returns:
        A dataframe containing spreadsheets data for classification.
    """
    dataframe_filepath = os.path.join(results_dir, 'dataframe.csv')

    if os.path.isfile(dataframe_filepath):
        print("A copy of the dataframe was found.")
        print("Reading the .csv file...")
        dataframe = pandas.read_csv(dataframe_filepath)
        dataframe.fillna('', inplace=True)
    else:
        print("No copies of the dataframe were found.")
        print("Parsing raw spreadsheets and preparing a new copy of the dataframe...")
        dataframe = parse_spreadsheets(analysis_dir, results_dir, classes)

    return dataframe

def parse_spreadsheets(analysis_dir, output_dir, classes):
    """Parses the spreadsheets and exports the dataframe as a unique file.

    Args:
        analysis_dir: A string representing the path to the directory where the spreadsheets 
                      are located.
        output_dir: A string representing the path to the directory where the dataframe
                     will be saved. If `output_dir` is empty, the dataframe will not be exported.
        classes: A list of strings containing the columns that should be extracted from each spreadsheet
            as classes of the classifier.

    Returns:
        A pandas dataframe with all the spreadsheets data into a single dataframe structure.
    """
    dataframe = pandas.DataFrame()

    for filename in os.listdir(analysis_dir):
        filepath = os.path.join(analysis_dir, filename)

        if os.path.isfile(filepath):
            # Notice that we consider any spreadsheets (.xlsx files)
            # inside the folder as valid for use.
            if filename.endswith('.xlsx'):
                worksheets = parse_spreadsheet_file(filepath, filename, classes)
                dataframe = pandas.concat([dataframe, worksheets])

    if output_dir:
        # Export as a .csv file:
        csv_filepath = os.path.join(output_dir, 'dataframe.csv')
        dataframe.to_csv(csv_filepath, index=False)
        # Export as a .xlsx file:
        excel_filepath = os.path.join(output_dir, 'dataframe.xlsx')
        dataframe.to_excel(excel_filepath, index=False, sheet_name="data")

    return dataframe

def parse_spreadsheet_file(filepath, classes):
    """Extracts the data from one spreadsheet file.

    Args:
        filepath: A string representing the path to a spreadsheet.
        classes: A list of strings containing the columns that should be extracted from the spreadsheet
            as classes of the classifier.

    Returns:
        A pandas dataframe including the data from all worksheets inside the spreadsheet.
    """
    spreadsheet = pandas.ExcelFile(filepath, engine='openpyxl')
    dataframe = pandas.DataFrame()

    for worksheet_name in spreadsheet.sheet_names:
        worksheet = spreadsheet.parse(worksheet_name)

        # Add a new name to the first column
        worksheet.rename(columns={worksheet.columns[0]: "Paragraph" }, inplace = True)
        # Replace NaN's with 0's and non NaN's with 1's
        for _class in classes:
            if _class in worksheet:
                worksheet[_class] = worksheet[_class].notnull().astype('int')
                
        worksheet['Label'] = worksheet.apply(lambda row: define_label(row, classes), axis=1)
        worksheet['Spreadsheet'] = os.path.basename(filepath)
        worksheet['Worksheet'] = worksheet_name
        worksheet['Row Index'] = worksheet.index
        dataframe = pandas.concat([worksheet, dataframe])

    return dataframe

def define_label(row, classes):
    """Identifies which label should be assigned to a row in a spreadsheet

    This study solves a multiclass classification problem. For this reason, a column
    named as `label` is created for each row in a spreadsheet file. This method identifies
    which one of the columns representing classes of the problem is the one that should be assigned
    as a label for the respective respective row. 

    Args:
        row: A pandas dataframe row. 
        classes: A list of strings containing the columns that represent the classes of the classifier
        in the dataframe.
    
    Returns:
        A string representing the label for the respective row. Notice that this string must be one
        of the strings contained in the classes parameter or the 'No categories identified.' string.
    """
    label = 'No categories identified.'

    for _class in classes:
        if _class in row:
            if row[_class] == 1:
                label = _class

    return label