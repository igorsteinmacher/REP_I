from scraper.validate import validate_documentation

# In our study, we filter the repositories based on their documentation files
# before manually analyzing them. We ignore repositories that contain empty 
# documentation files, files not written in English or not written in Markdown.

def test_incomplete_file():
    """Tests for incomplete documentation files
    """
    documentation_files = [{'filename': 'README',
                            'content': None,
                            'description': None}]

    is_valid, reasons_for_invalidation = validate_documentation(documentation_files)

    assert is_valid is False
    assert "README has missing values." in reasons_for_invalidation

def test_incomplete_description_file():
    """Tests for incomplete description fields
    """
    documentation_files = [{'filename': 'README',
                            'content': 'Hello world!',
                            'description': {'name': 'README.md'}}]

    is_valid, reasons_for_invalidation = validate_documentation(documentation_files)

    assert is_valid is False
    assert "README has missing values." in reasons_for_invalidation

def test_empty_file():
    """Tests for empty files
    """
    documentation_files = [{'filename': 'README',
                            'content': 'Hello world!',
                            'description': {'name': 'README.md',
                                            'size': 1}}]

    is_valid, reasons_for_invalidation = validate_documentation(documentation_files)

    assert is_valid is False
    assert "README size < 0.5kB." in reasons_for_invalidation

def test_file_not_in_markdown():
    """Tests for files not written in Markdown
    """
    documentation_files = [{'filename': 'README',
                            'content': 'Hello world!',
                            'description': {'name': 'README.rst',
                                            'size': 500}}]

    is_valid, reasons_for_invalidation = validate_documentation(documentation_files)

    assert is_valid is False
    assert "README is not in Markdown." in reasons_for_invalidation

def test_file_not_in_english():
    """Tests for files not written in English
    """
    documentation_files = [{'filename': 'README',
                            'content': '你好世界',
                            'description': {'name': 'README.md',
                                            'size': 500}}]

    is_valid, reasons_for_invalidation = validate_documentation(documentation_files)

    assert is_valid is False
    assert "README is not in English." in reasons_for_invalidation


def test_valid_file():
    """Tests for valid documentation files
    """
    documentation_files = [{'filename': 'README',
                            'content': 'Hello world!',
                            'description': {'name': 'README.md',
                                            'size': 500}}]

    is_valid, reasons_for_invalidation = validate_documentation(documentation_files)

    assert is_valid is True
    assert len(reasons_for_invalidation) <= 1