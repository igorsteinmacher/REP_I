#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
import xlrd
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

def parse_spreadsheets(analysis_dir, output_dir, desired_worksheets = ['contributing']):
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
    dataframe_rows = []

    for filename in os.listdir(analysis_dir):
        filepath = os.path.join(analysis_dir, filename)

        if os.path.isfile(filepath):
            # Notice that we consider any spreadsheets (.xlsx files)
            # inside the folder as valid for training.
            if filename.endswith('.xlsx'):
                spreadsheet = parse_spreadsheet_file(filepath, desired_worksheets, filename)

                for worksheet in spreadsheet:
                    dataframe_rows = dataframe_rows + spreadsheet[worksheet]

    dataframe = pandas.DataFrame(dataframe_rows)

    if output_dir:
        # Export as a .csv file:
        csv_filepath = os.path.join(output_dir, 'dataframe.csv')
        dataframe.to_csv(csv_filepath, index=False)
        # Export as a .xlsx file:
        excel_filepath = os.path.join(output_dir, 'dataframe.xlsx')
        dataframe.to_excel(excel_filepath, index=False, sheet_name="data")

    return dataframe

def parse_spreadsheet_file(filepath, desired_worksheets, spreadsheet):
    """Extracts the data from a spreadsheet file.

    Args:
        filepath: A string representing the path to a spreadsheet.
        desired_worksheets: A list of worksheets to be parsed from this spreadsheet.
        spreadsheet: A string representing the name of the spreadsheet being analyzed.

    Returns:
        A dictionary including all parsed worksheets. The set of rows in each
        worksheet is now represented by a list of dictionaries.
    """
    spreadsheet = xlrd.open_workbook(filepath)
    worksheets = {}

    for worksheet_name in desired_worksheets:
        worksheet = spreadsheet.sheet_by_name(worksheet_name).get_rows()
        column_names = next(worksheet)
        parsed_rows = []

        for index, row in enumerate(worksheet):
            # The method below returns the values of each column
            # as a dictionary of columns
            row_data = parse_column_values(row, column_names)
            # We include some extra values for each row that might be used
            # in the future during the data analysis:
            row_data['Spreadsheet'] = spreadsheet
            row_data['Worksheet'] = worksheet_name
            row_data['Row Index'] = index + 2

            parsed_rows.append(row_data)

        worksheets[worksheet_name] = parsed_rows

    return worksheets

def parse_column_values(row, column_names):
    """Parses the column values of a row and turns it into a dictionary.

    IMPORTANT: Notice that this method is specifically used for spreadsheets following 
    the standards of our study where each worksheet has six columns such that
    the first one always contains a paragraph/text, and the remaining five columns
    contain markers identifying one of the desired categories of our study.

    If you are going to use this code in another study, please modify it as necessary.

    Args:
        row: A list containing the row values of a worksheet.
        column_names: A list with the respective classes/categories of this study.

    Returns:
        The worksheet row in form of dictionary.
    """
    
    row_dict = {}
    # The first column is always the paragraph
    row_dict['Paragraph'] = row[0].value

    # The remaining columns represent the classes
    # to be used in the multiclass classification.
    for column in range(1, len(column_names)):
        # If there are any markers in the respective column:
        if row[column].ctype == 1:
            row_dict[column_names[column].value] = 1
            row_dict['label'] = column_names[column].value
        else:
            row_dict[column_names[column].value] = 0

    return row_dict
