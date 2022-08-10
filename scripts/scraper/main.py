#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
import logging
from datetime import datetime
from scrap import scrap_documentation_file, scrap_repositories
from export import create_analysis_file, export_to_repositories_file
from validate import validate_documentation

def scrap_validate_and_export(programming_languages, api_pages, output_dir):
    """Performs the steps of scraping, validating and exporting data and documentation.

    In our study, we analyze qualitatively the documentation files of popular open
    source repositories hosted on GitHub, in order to identify what information
    are relevant to new contributors. To achieve this objective, we first extract 
    the documentation files of these projects from GitHub. We remove files that
    are invalid (for example, files not written in English), and we export these
    documentation files as spreadsheets for manual analysis. This method peforms,
    in sequence, all the necessary steps of this first objective of our study.

    Args:
        programming_languages: A list of strings representing programming languages
            of which the most popular repositories will be extracted.
        api_pages: A list of integers representing the pages to be extracted for
            each programming language on GitHub API.
        output_dir: A string representing the directory path where the data and
            files about the extracted repositories will be saved.
    """

    repositories_filepath = os.path.join(output_dir, 'repositories.csv')

    # (1) Creates the output directory, if it does not exist.
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # (2) Creates a directory where the projects files for analysis will be saved, 
    # if it does not exist.

    analysis_dir = os.path.join(output_dir, 'documentation-spreadsheets')

    if not os.path.isdir(analysis_dir):
        os.makedirs(analysis_dir)

    # (3) Extract most popular repositories from GitHub API. 

    repositories = scrap_repositories(programming_languages, api_pages)

    for index, repository in enumerate(repositories):
        try:
            # (4) For each repository collected, extract the `CONTRIBUTING.md`
            # documentation file.

            owner, name = repository['owner']['login'], repository['name']
            contributing = scrap_documentation_file(owner, name, 'contributing')

            # (5) Check if the documentation file is valid (attend the requirements).

            is_valid, reasons_for_invalidation = validate_documentation(contributing)

            # (6) If the documentation file is valid, create a Markdown file, and save it
            # into a `raw` folder, inside the output directory.

            raw_dir = os.path.join(output_dir, 'documentation-raw')

            if not os.path.isdir(raw_dir):
                os.makedirs(raw_dir)

            if is_valid:
                filename = owner + '@' + name + '.txt'
                raw_filepath = os.path.join(raw_dir, filename)

                with open(raw_filepath, 'w', errors='replace') as writer:
                    writer.write(contributing['content'])
                    writer.close()

                # (7) For each project containing a valid documentation file, create 
                # a spreadsheet containing the paragraphs of this documentation file
                # to be used in qualitative analysis, and export this spreadsheet
                # to the `analysis` folder, inside the output directory.

                spreadsheet_filepath = os.path.join(analysis_dir, owner + '@' + name + '.xlsx')
                create_analysis_file('contributing', raw_filepath, spreadsheet_filepath)

            # (8) Update the dictionary of the collected repository only with the
            # necessary information, including the is_valid flag and the possible
            # reasons for invalidation.

            repository_information = {
                'id': repository['id'],
                'owner': owner,
                'name': name,
                'url': repository['html_url'],
                'language': repository['language'],
                'description': repository['description'],
                'extracted_at': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                'is_valid': is_valid,
                'reasons_for_invalidation': reasons_for_invalidation
            }

            repositories[index] = repository_information

            # (9) Export the repository information to the `repositories.csv` spreadsheet,
            # inside the output folder.

            export_to_repositories_file(repository_information, repositories_filepath)

        except Exception as exception:
            # Attention:
            # Sometimes when we scrap GitHub projects it is hard to
            # predict what kind of weird data they will return.
            # To prevent the scraping process from stopping, I use this
            # generic try/catch in main.py. However, I highly recommend you,
            # developer, to review the exceptions.log after every execution.
 
            logging.basicConfig(filename='exceptions.log', level=logging.DEBUG)
            logging.info('Generic exception caught in main.py')
            logging.exception(exception)

    return repositories

if __name__ == '__main__':
    root_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(root_dir, 'data')

    # Reference to justify why we are using these programming languages: octoverse.github.com (or see misc/octoverse-top-languages.png)
    programming_languages = ['JavaScript', 'Python', 'Java', 'PHP', 'C#', 'C++', 'TypeScript', 'Shell', 'C', 'Ruby']
    api_pages = [i for i in range(1, 35)]
    scrap_validate_and_export(programming_languages, api_pages, data_dir)
