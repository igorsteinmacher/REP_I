#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
import xlrd
import pandas


def check_if_dataframe_copy_exists(results_dir, analysis_dir):
    """Checks if there is a csv file available to be used as dataframe.

    To avoid parsing the spreadsheets in every execution (which takes time), a
    copy of the dataframe will be saved inside the `results/csv/` directory. If the
    copy already exists, it will be converted to a dataframe by Pandas. If not,
    the parsing methods will be executed and the copy will be prepared.

    Args:
        results_dir: A string path to the directory where the dataframe may be saved.
        analysis_dir: A string path to the analysis directory containing the
                        students spreadsheets.

    Returns:
        A dataframe containing spreadsheets data for classification.
    """
    csv_dir = os.path.join(results_dir, 'csv')
    raw_dataframe_filepath = os.path.join(csv_dir, 'raw_dataframe.csv')

    if os.path.isfile(raw_dataframe_filepath):
        print("Reading dataframe file for classification.")
        print("Filepath:" + raw_dataframe_filepath)
        dataframe = pandas.read_csv(raw_dataframe_filepath)
    else:
        print("Parsing spreadsheets for classification.")

        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)

        dataframe = parse_spreadsheets(
            analysis_dir, ['README', 'CONTRIBUTING'], csv_dir)

    return dataframe


def parse_spreadsheets(analysis_dir, desired_worksheets, output_dir):
    """Parses students spreadsheets from the traning directory for classification.

    Args:
        analysis_dir: A string path the to directory where the students
                        spreadsheets are located.
        desired_worksheets:  List of worksheet names to be extracted.
        output_dir: A string path to the directory where the resulting dataframe
                     will be saved. If empty, the dataframe will not be exported.

    Returns:
        A dataframe grouping all the students spreadsheets data into a single
        structure.

    Notes:
        The subdirectories inside the traning directory must comprehend the
        following structure:

        training
        └── university
            └── student
                └── student_spreadsheet.xlsx

        Where "university" is a directory that represents a college where the
        experiment was applied, "student" is a directory representing a participant
        of the experiment in that university, and student_spreadsheet.xlsx is a
        spreadsheet analyzed by the respective student.
    """
    dataframe_rows = []

    for university in os.listdir(analysis_dir):
        for author in os.listdir(os.path.join(analysis_dir, university)):
            for filename in os.listdir(os.path.join(analysis_dir, university, author)):
                filepath = os.path.join(
                    analysis_dir, university, author, filename)
                if os.path.isfile(filepath):
                    if filename.endswith('.xlsx'):
                        extra_params = {'University': university,
                                        'Author': author, 'Filename': filename}
                        spreadsheet = parse_spreadsheet(
                            filepath, desired_worksheets, extra_params)

                        for worksheet in spreadsheet:
                            dataframe_rows = dataframe_rows + \
                                spreadsheet[worksheet]

    dataframe = pandas.DataFrame(dataframe_rows)

    if output_dir:
        filepath = os.path.join(output_dir, 'raw_dataframe.csv')
        dataframe.to_csv(filepath, index=False)

    return dataframe


def parse_spreadsheet(filepath, desired_worksheets, extra_params=[]):
    """Extracts data from a student spreadsheet.

    Args:
        filepath: A string path to the spreadsheet.
        desired_worksheets: A list of worksheets to be parsed.
        extra_params: A list of parameters to add in each generated row
                    dictionary.

    Returns:
        A dictionary including all parsed worksheets. The set of rows in each
        worksheet is now represented by a list of dictionaries.
    """
    spreadsheet = xlrd.open_workbook(filepath)
    worksheets = {}

    for worksheet_name in desired_worksheets:
        worksheet = spreadsheet.sheet_by_name(worksheet_name).get_rows()
        rows = []
        column_names = next(worksheet)
        for index, row in enumerate(worksheet):
            row_dict = parse_row(row, column_names)
            row_dict.update(extra_params)
            row_dict['Paragraph Index'] = index + 2
            row_dict['Document'] = worksheet_name
            rows.append(row_dict)
        worksheets[worksheet_name] = rows

    return worksheets


def parse_row(row, column_names):
    """Parses the columns of a worksheet row and turns it into a dictionary.

    Args:
        row: A list containing the row values of a worksheet.
        column_names: A list with the respective names for each column.

    Returns:
        The worksheet row in form of dictionary.
    """
    row_dict = {}
    row_dict['Paragraph'] = row[0].value

    for column in range(1, len(column_names)):
        if row[column].ctype == 1:
            row_dict[column_names[column].value] = 1
        else:
            row_dict[column_names[column].value] = 0
    return row_dict
