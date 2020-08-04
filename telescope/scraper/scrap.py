#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ =  'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import api_scraper as scraper

def scrap_repositories(programming_languages, api_pages):
    """Scraps repositories hosted on GitHub, ordered by popularity and language.

    This method uses GitHub API to collect the most popular repositories hosted
    publicly on GitHub for a list of programming languages. The popularity of the
    repositories are measured by their number of stars (H. Borges, ICSME 2016).
    For each programming language, `len(api_pages) * 30` projects are extracted.
    The pages are ordered by popularity, starting at page one. 

    Args:
        programming_languages: A list of strings representing programming languages
            of which the most popular repositories will be extracted.
        api_pages: A list of integers representing the pages to be extracted for
            each programming language.
    References:
        Borges, Hudson, Andre Hora, and Marco Tulio Valente. "Understanding
        the factors that impact the popularity of GitHub repositories." 2016
        IEEE International Conference on Software Maintenance and Evolution
        (ICSME). IEEE, 2016.
    """
    api_scraper = scraper.Create()
    api_repositories_url = 'https://api.github.com/search/repositories'
    repositories = []

    for language in programming_languages:
        for page in api_pages:
            print("Extracting repositories in page {} written in {}.".format(page, language))
            parameters = {'q': 'language:' + language, 'sort': 'stars', 'order': 'desc', 'page': page}
            response = api_scraper.request(api_repositories_url, parameters)['items']

            # Some projects received from API have a None value for the language
            # parameter instead of their respective language, so in such cases
            # we manually update the repository's language.

            for index, repository in enumerate(response):
                repository['language'] = language
                response[index] = repository

            repositories = repositories + response

    return repositories

def scrap_documentation_file(owner, name, filename):
    """Scraps a documentation file of a repository hosted on GitHub.

    Args:
        owner: String representing the organization or user owner of the repository.
        name: String representing the repository name.
        filename: The name of the documentation file that will be extracted.
    Returns:
        A dictionary containing three values: the name of the extracted file, 
        the description of this file (represented by the 'description' key),
        and the content of the file (represented by the 'content' key). If
        one of the last two values are not found in the API, None is returned.
    """

    api_scraper = scraper.Create()
    documentation_file = {'filename': filename, 'content': None, 'description': None}

    # The community profile is used to get a documentation file of a repository. 
    # The definition of community profile is available at the API documentation:
    # developer.github.com/v3/repos/community. 
    # Notice that to request the community profile of a project, it is necessary
    # to add in the request header the flag defined below.

    flag = 'application/vnd.github.black-panther-preview+json'   
    community_profile_url = 'https://api.github.com/repos/{}/{}/community/profile'.format(owner,name)
    community_profile = api_scraper.request(community_profile_url, headers={'Accept': flag})

    # In some community profiles, the necessary values are missing, and we can
    # not predict it. For this reason, we need to check if all the keys and values
    # exist before performing the scraping of the documentation file.

    if community_profile:
        if 'files' in community_profile:
            if filename in community_profile['files']:
                if community_profile['files'][filename] is not None:
                    if 'url' in community_profile['files'][filename]:
                        description_url = community_profile['files'][filename]['url']
                        description = api_scraper.request(description_url)
                        documentation_file['description'] = description

                        if description:
                            if 'download_url' in description:
                                if description['download_url'] is not None:
                                    print("Downloading documentation files of {}/{}.".format(owner, name))
                                    download_url = description['download_url']
                                    content = api_scraper.request(download_url, file_type='text')
                                    documentation_file['content'] = content
        
    return documentation_file