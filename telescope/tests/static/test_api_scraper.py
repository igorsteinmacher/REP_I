import time
from datetime import datetime, timedelta
from scraper.api_scraper.scraper import Create

github_scraper = Create()

def test_repositories_returned_values():
    """Checks whether the values of the repositories ​​returned by GitHub API match
    those used in Telescope.

    Suppose that GitHub developers decide to change the field name `owner` of a
    repository by `owners`. This change would create an inconsistency in the 
    execution of Telescope. For this reason, this method performs a set of asserts
    to ensure that the values extracted from the GitHub API are still the same.  
    """
    parameters = {'q': 'language:Java', 'sort': 'stars', 'order': 'desc'}
    repositories = github_scraper.request('https://api.github.com/search/repositories', parameters)

    for repository in repositories['items']:
        assert repository['id'] is not None, "Field `id` is None."
        assert repository['owner'] is not None, "Field `owner` is None."
        assert repository['name'] is not None, "Field `name` is None."
        assert repository['url'] is not None, "Field `url` is None."
        assert repository['language'] is not None, "Field `language` is None."
        assert repository['description'] is not None, "Field `description` is None."

    assert len(repositories['items']) == 30, "Number of repositories is different than 30."

def test_documentation_files_returned_values():
    """Checks whether the values of the documentation files of a repository ​​returned
    by GitHub API match those used in Telescope.

    Suppose that GitHub developers decide to change the field name `files` of a
    repository by `docfiles`. This change would create an inconsistency in the 
    execution of Telescope. For this reason, this method performs a set of asserts
    to ensure that the values extracted from the GitHub API are still the same.

    This method also extracts the documentation file of the repository used
    as example as a text file, and confirms if the extracted file is valid.
    """

    # The community profile is used to get a documentation file of a repository. 
    # The definition of community profile is available at the API documentation:
    # developer.github.com/v3/repos/community. 
    # Notice that to request the community profile of a project, it is necessary
    # to add in the request header the flag defined below (that will also be tested).

    flag = 'application/vnd.github.black-panther-preview+json' 
    community_profile_url = 'https://api.github.com/repos/ruby/ruby/community/profile'
    community_profile = github_scraper.request(community_profile_url, headers={'Accept': flag})

    assert community_profile is not None, "Request `community/profile' is None."
    assert community_profile['files'] is not None, "Field `files` is None."
    assert community_profile['files']['readme']['url'] is not None, "Field `files -> readme -> url` is None."
    assert community_profile['files']['contributing']['url'] is not None, "Field `files -> readme -> url` is None."

    # Requests the description of the README.md file

    description = github_scraper.request(community_profile['files']['readme']['url'])
    
    assert description['download_url'] is not None

    # Requests the raw version of the README.md file available at the repository

    raw_content = github_scraper.request(description['download_url'], file_type='text')

    assert len(raw_content) > 0


def test_rate_limit_verification():
    """Verifies if the `verify_rate_limit` method suspends the scraping process

    When the maximum number of requests available is reached, the `verify_rate_limit`
    method must suspend the scraping process until the reset datetime is reached.
    This method asks the `verify_rate_limit` to suspend the process for five seconds
    and tests if the sleeping process worked or not. 
    """
    start_of_execution = time.time()
    reset_datetime = datetime.today() + timedelta(0, 5)
    reset_timestamp = datetime.timestamp(reset_datetime)
    dummy_header = {'X-RateLimit-Remaining': 0, 'X-RateLimit-Reset': reset_timestamp}
    github_scraper.verify_rate_limit(dummy_header)
    end_of_execution = time.time()
    assert (end_of_execution - start_of_execution) > 5, "The process was not suspended by `verify_rate_limit`."
