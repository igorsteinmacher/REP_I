import streamlit as page
from about_section import write_about_section
from analysis_section import write_contributing_analysis

page.title("contributing.info")

repository_url = page.text_input("What GitHub repository would you like to\
                 analyze?", help="Please provide a valid URL to the repository\
                 hosted on GitHub that you would like to analyze.", 
                 placeholder="https://github.com/github/docs/", max_chars=2048)

with page.spinner("Parsing documentation file..."):
    write_contributing_analysis(page, repository_url)

write_about_section(page)