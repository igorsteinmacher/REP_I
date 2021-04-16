#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ =  'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
from langdetect import detect

def validate_documentation(document):
    """Checks if the documentation file of a project meets the research set of requirements.

    To avoid files that may not be valid for qualitative analysis, a set of
    filters are applied in the documentation files of each project. In other 
    words, the documentation file of a project is considered valid if the data
    extracted for is complete (i.e. the values returned from the GitHub API are 
    valid), the documentation file is not empty (i.e. size greater or equal to
    0.5kB), and the file written in English and in Markdown format. Projects
    with documentation files that do not fit these requirements receive as output
    an invalidation flag with the respective reasons for their invalidation.

    Args:
        document: Data of a documentation file of a project, in accordance with the 
            scrap_documentation_file method output.
    Returns:
        A boolean value representing the invalidation flag is_valid and a string
        containing the possible reasons for invalidation. The flag is_valid is True
        if the document data provided is valid, and is_valid is False otherwise. 
        If the documentation file is valid, the variable reasons_for_invalidation
        will be an empty string. 
    """  

    def check_if_is_complete(document):
        nonlocal is_valid
        if document['content'] is None or document['description'] is None:
            is_valid = False
            reasons_for_invalidation.append(document['filename'] + " is missing.")
        else:
            if 'size' not in document['description'] or 'name' not in document['description']:
                is_valid = False
                reasons_for_invalidation.append(document['filename'] + " is missing.")

    def check_if_is_empty(document):
        nonlocal is_valid
        file_size = document['description']['size'] # in Bytes

        if file_size < 500:
            is_valid = False
            reasons_for_invalidation.append(document['filename'] + " size < 0.5kB.")

    def check_if_is_written_in_markdown(document):
        nonlocal is_valid
        filename = document['description']['name']
        file_extension = os.path.splitext(filename)[1].lower()

        supported_extensions = ['.markdown', '.mdown', '.mkdn', '.md']
        if file_extension not in supported_extensions:
            is_valid = False
            reasons_for_invalidation.append(document['filename'] + " is not in Markdown.")

    def check_if_is_written_in_english(document):
        nonlocal is_valid
        file_language = detect(document['content'])

        if file_language != 'en':
            is_valid = False
            reasons_for_invalidation.append(document['filename'] + " is not in English.")

    is_valid = True # Flag that defines if the documentation file is valid
    reasons_for_invalidation = [] # List of strings definining reasons for file invalidation

    # First, we check if the data of the documentation file
    # is available in the GitHub API.

    check_if_is_complete(document)

    # If it is available, we check if the documentation file
    # is empty, written in Markdown and in English.

    if is_valid:
        check_if_is_empty(document)
        check_if_is_written_in_markdown(document)
        check_if_is_written_in_english(document)
        
    return is_valid, '\n'.join(reasons_for_invalidation)