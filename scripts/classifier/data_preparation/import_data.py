import os
from data_preparation.prepare_data import create_train_and_test_sets, import_sets

def import_data_for_classification(spreadsheets_dir, data_dir):
    # Spreadsheets headers
    text_column = 'Paragraph'   
    classes_columns = ['No categories identified.',
                       'CF – Contribution flow',
                       'CT – Choose a task',
                       'TC – Talk to the community',
                       'BW – Build local workspace',
                       'DC – Deal with the code',
                       'SC – Submit the changes']

    # Label for a new column header that will merge
    # classes_columns into a single column
    label_column = 'Label'

    # Filepaths where the train and test sets are saved
    train_filepath = os.path.join(data_dir, 'train.csv')
    test_filepath = os.path.join(data_dir, 'test.csv')

    if not os.path.exists(train_filepath) or not os.path.exists(test_filepath):
        create_train_and_test_sets(spreadsheets_dir, text_column, 
                                   classes_columns, train_filepath, test_filepath,
                                   label_column, data_dir)

    return import_sets(train_filepath, test_filepath, text_column, label_column)