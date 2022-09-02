# ICSE-2023
[![DOI](https://zenodo.org/badge/202890143.svg)](https://zenodo.org/badge/latestdoi/202890143)

Welcome to the replication package for the paper entitled "How do CONTRIBUTING files support newcomers in Open Source projects?", submitted to ICSE 2023. 

# Repository Structure 
Our package is divided in folders and it is organized as follows:
- app: This folder contains the implementation of our streamlit application, used to demonstrate the capabilities of our classification model. 
- data: If you are looking for the data we have used for classification, this is the folder where it is located. This folder also contains the raw contributing files of each project analyzed, and the spreadsheets qualitatively analyzed by the authors. 
- results: This folder contains all the files related to our results section. From the analysis of our classification model to the results of our online questionnaire.
- scripts: If you want to see the code we wrote during the whole classification process, including but not limited to the classification process, this is the folder you are looking for.
- qualification: This folder contains the first set of data we tried to use to train our classifier. This data, analyzed by undegraduate students, was part of my masters qualification and wans't used or discussed in this paper. I just keep it here for recording purposes. 
- [misc: This folder contains miscellaneous files that were not used in this paper, but support the statements of it (e.g. a screenshot of the top ten languages used on GitHub from the Octoverse website).

# I want to use your model!
We are glad you are interested in our classification model. The final model is available as a `classification_model.sav` file inside the `app` folder that you can load using Pickle. If you are not familiar with Pickle or don't know how to load a model, we recommend you to take a look at our code implementation inside the `app` folder (`classify_content.py` is a good starting point).
