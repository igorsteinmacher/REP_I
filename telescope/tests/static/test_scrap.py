from scraper.scrap import scrap_repositories, scrap_documentation_file

def test_repositories_scraping():
    """Checks if the method of scraping repositories returns the correct values

    The GitHub API scraper is tested in `test_api_scraper.py`. This is a complementary
    testing method that validates the tasks executed by the `scrap_repositories` 
    function.
    """
    programming_languages = ['JavaScript']
    api_pages = [1]
    repositories = scrap_repositories(programming_languages, api_pages)

    for repository in repositories:
        assert repository['language'] == 'JavaScript'

    assert len(repositories) == 30

def test_documentation_scraping():
    """Checks if the method of scraping documentation files returns the correct values

    This method validates the `scrap_documentation_file` function by using the
    Ruby project as example. It scraps the `readme` file inside the Ruby repository
    and confirms that the fields returned by the scraping method are available.
    """
    owner = 'ruby'
    name = 'ruby'
    filename = 'readme'
    documentation_file = scrap_documentation_file(owner, name, filename)
    
    assert 'filename' in documentation_file
    assert 'content' in documentation_file
    assert 'description' in documentation_file
