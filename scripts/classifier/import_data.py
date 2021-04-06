#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
import pandas


def import_dataframe(analysis_dir, results_dir):
    """Imports the data that will be used for training as a pandas dataframe.
   
    First of all, it is important to understand that the data used in our study
    is primarily represented by a folder of spreadsheets which were manually analyzed
    by software engineering researchers. These spreadsheets follow a same standard:
    each contains a single worksheet named as `contributing` with seven pre-defined columns
    and a variable number of rows.

    To avoid parsing all these spreadsheets in every execution (which takes time and memory),
    we parse the spreadsheets at the first execution and save a xlsx copy of the dataframe 
    inside the `results_dir` directory.

    If this copy already exists, the xlsx file is directly converted to a pandas dataframe. 
    Otherwise, the parsing methods are executed on the spreadsheets available inside the
    `analysis_dir` and a new copy of the dataframe is prepared.

    Args:
        analysis_dir: A string representing a path to the directory containing the 
                      spreadsheets used during the qualitative analysis.
        results_dir: A string representing a path to the directory where the 
                     dataframe is/will be saved.

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
        dataframe = parse_spreadsheets(analysis_dir, results_dir)

    return dataframe

def parse_spreadsheets(analysis_dir, output_dir):
    """Parses the spreadsheets and exports the dataframe as a unique file.

    Args:
        analysis_dir: A string representing the path to the directory where the raw spreadsheets 
                      are located.
        output_dir: A string representing the path to the directory where the dataframe
                     will be saved. If `output_dir` is empty, the dataframe will not be exported.
        desired_worksheets:  List of worksheet names to be extracted. The default is ['contributing'].

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
                worksheets = parse_spreadsheet_file(filepath, filename)
                dataframe = pandas.concat([dataframe, worksheets])

    if output_dir:
        # Export as a .csv file:
        csv_filepath = os.path.join(output_dir, 'dataframe.csv')
        dataframe.to_csv(csv_filepath, index=False)
        # Export as a .xlsx file:
        excel_filepath = os.path.join(output_dir, 'dataframe.xlsx')
        dataframe.to_excel(excel_filepath, index=False, sheet_name="data")

    return dataframe

def parse_spreadsheet_file(filepath, spreadsheet):
    """Extracts the data from a spreadsheet file.

    Args:
        filepath: A string representing the path to a spreadsheet.
        spreadsheet: A string representing the name of the spreadsheet being analyzed.

    Returns:
        A dictionary including all parsed worksheets. The set of rows in each
        worksheet is now represented by a list of dictionaries.
    """
    spreadsheet = pandas.ExcelFile(filepath, engine='openpyxl')
    dataframe = pandas.DataFrame()

    for worksheet_name in spreadsheet.sheet_names:
        worksheet = spreadsheet.parse(worksheet_name)

        # Add a new name to the first column
        worksheet.rename(columns={worksheet.columns[0]: "Paragraph" }, inplace = True)
        # Replace NaN's with 0's and non NaN's with 1's
        worksheet['CF – Contribution flow'] = worksheet['CF – Contribution flow'].notnull().astype('int')
        worksheet['CT – Choose a task'] = worksheet['CT – Choose a task'].notnull().astype('int')
        worksheet['TC – Talk to the community'] = worksheet['TC – Talk to the community'].notnull().astype('int')
        worksheet['BW – Build local workspace'] = worksheet['BW – Build local workspace'].notnull().astype('int')
        worksheet['DC – Deal with the code'] = worksheet['DC – Deal with the code'].notnull().astype('int')
        worksheet['SC – Submit the changes'] = worksheet['SC – Submit the changes'].notnull().astype('int')
        worksheet['Label'] = worksheet.apply(lambda row: define_label(row), axis=1)
        worksheet['Spreadsheet'] = os.path.basename(filepath)
        worksheet['Worksheet'] = worksheet_name
        worksheet['Row Index'] = worksheet.index
        dataframe = pandas.concat([worksheet, dataframe])

    return dataframe

def define_label(row):
        label = 'No categories identified.'

        if row['CF – Contribution flow'] == 1:
            label = 'CF – Contribution flow'
        if row['CT – Choose a task'] == 1:
            label = 'CT – Choose a task'
        if row['TC – Talk to the community'] == 1:
            label = 'TC – Talk to the community'
        if row['BW – Build local workspace'] == 1:
            label = 'BW – Build local workspace'
        if row['DC – Deal with the code'] == 1:
            label = 'DC – Deal with the code'
        if row['SC – Submit the changes'] == 1:
            label = 'SC – Submit the changes'

        return label