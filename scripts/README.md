# About

This is probably the most complex folder of all the repository, so I will try to be as detailed as possible.

This folder is organized as follows:
- If you are looking for how we extracted documentation data from GitHub, you should look at the `scraper` folder. The `api_scraper.py` file is the main file of this folder, containing the code that requests custom URLs to GitHub API. The file `main.py` presents the whole process of extracting a documentation file, `scrapy.py` shows how to do the URL requets to the `api_scraper.py` module and `validate.py` shows how we validated if a documentation file was valid for qualitative analysis or not. If you want to know how we converted the markdown files to spreadsheets, take a look at `export.py` (noticed that we use cmark-gfm to convert the markdown content to plaintext, which might be a pain if you are not using a system based on Linux). More information about all these files are given as doctstrings. 
