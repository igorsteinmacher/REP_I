import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="telescope",
    version="0.0.1",
    author="Felipe Fronchetti",
    author_email="fronchetti@usp.br",
    description="The set of modules used in Telescope",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fronchetti/USP-2020",
    packages=['classifier', 'scraper', 'scraper.api_scraper', 'utils'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
