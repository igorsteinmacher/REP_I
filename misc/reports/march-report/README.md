# March Report
After the last [meeting](https://github.com/fronchetti/IME-USP/tree/master/first_analysis/meeting-19-02-2019), we defined our 
methodology based on the following steps:

1) **Define the categories we want to predict**:
We will review the Barriers Model and define the categories we want to identify in documentation files.
Since some barriers are not related to documentation files (e.g. Shyness), we will consider as categories only barriers that
can be textually described in a documentation file (e.g. How to submit a pull request).

2) **Write a tutorial on how to (qualitatively) code the documentation files**:
With the categories defined, we will write a tutorial on how to identify 
the categories in README and CONTRIBUTING files.
This tutorial will guide future qualitative analyzes. The idea is to 
describe each category giving examples from OSS projects already qualified.

3) **Conduct a pilot analysis**:
Steinmacher, Wiese, Christoph, Gustavo and Felipe, using the tutorial 
described in step two, will code the documentation files.
They will also describe, for each category, where and why they used the 
category. At the end, we will calculate the agreement using IRR, and if we 
find a satisfatory result, we will proceed to the next step. 

4) **Conduct the analysis with undergraduate students**:
Students from UFPA (Gustavo's University) and USP (Felipe's University) 
will execute the same codification made in step three.
They will follow our tutorial, and will code the categories for a set of OSS projects. The 
codes will be analyzed by the researchers, and then be used
as samples in the machine learning algorithm.

5) **Apply natural language processing techniques to predict categories**:
With a training set containing samples manually classified by the students,  we will make use of techniques in Natural Language Processing (NLP) to 
automatically identify the categories in a test set. The machine learning 
algorithm will learn with the samples, and will try to identify the same 
categories in new documentation files.

6) **Validate the results**: <br>
   We intend to validate the results using two different approaches:<br>
    i) Use students to evaluate the quality of the prediction:<br>
        We will ask students in a survey if they agree with the categories the algorithm predicted.<br>
        (e.g. "Is this paragraph related to the pull requests submission process?")<br>
    ii) Use students to select the best documentation:<br>
        We will identify the categories in documentation files and reorganize the categories into a new documentation file.<br>
        We will ask students which documentation file they prefer, the original or the reorganized one.<br>

For now on, we will report the progress based on these six steps.

## Step 1:
### Define the categories we want to predict
After discussing with Steinmacher, Wiese and Gustavo about the categories we want to predict, we decided to make some small changes in the categories we have defined in the last month. We kept with the same number of categories, but the definition for each one was modified to better fit the Barriers Model. The new set of categories is: 

* **CF – Contribution flow**:
It is not uncommon for newcomers to feel lost and unmotivated when it is not clear how to contribute with an OSS project. For this category, we would like to identify documentation that describes what is the contribution flow of a project. The contribution flow can be defined as a set of steps a newcomer need to follow to develop an acceptable contribution for the project.

* **CT – Choose a task**:
Many developers are interested in contributing with OSS projects, but most of them don’t know what task they should start with. In this category, we would like to identify sentences describing how newcomers can find a task to contribute with the project.

* **FM – Find a mentor**:
In most OSS projects, mentors are assigned to help newcomers during their contribution process. For this category, we would like to identify information about how newcomers can find people in the project to act as mentors during their contribution process.

* **TC – Talk to the community**:
Besides having a mentor, it is also important for newcomers to get in touch with the project’ community itself. For this reason, we would like to identify any information that describes how a newcomer can get in touch with the community members, including links for communication channels, tutorials on how to send a message, communication etiquette, etc.

* **BW – Build local workspace**:
Newcomers reported in previous studies that they did not find explanations about how they could build their own workspace (build, compile, execute, manage dependencies, etc.) before contributing. We would like to identify sections in documentation that defines the steps on how to build a local workspace.

* **DC – Deal with the code**:
Many projects have their own code standards, software architectures and practices. In this category, we would like to identify sections in documentation describing how code should be written, organized and documented.

* **SC – Submit the changes**:
The last step in the contribution process is the submission of changes. In this category, we would like to identify information about how the change/patch submission should be made.

## Extra Step:
### How we plan to conduct the experiment:
One of the most important things that we worked on this month and that was not listed in the steps, was to decide how the experiment will be conducted with the undergraduate students. First, we decided which projects we will use in our dataset. After some discussion, we decided to follow the Guilherme Avelino et. al paper about [Truck Factor in Popular Github Applications](https://peerj.com/preprints/1233.pdf), selecting projects that the paper provided with truck factor less or equal to three. We restricted our dataset for projects with both documentation files, README and CONTRIBUTING.

Secondly, we decided how the information will be arranged for codification: 
* For each project, we will have a spreadsheet file with two pages (i.e. two worksheets), one for the README and one for the CONTRIBUTING file. 
* Each page will contain eight columns:
  *  The first column will represent the text available in the respective documentation file, splitted into paragraphs. 
  * From the second to the sixth column, we will have empty columns, each one representing a category we want to identify.
  
The students will open a spreadsheet and, for each worksheet, they will read the paragraphs in the first column. If they identify a category in a paragraph row, they will mark in the same row an X symbol in the respective column of the category that they identified. The process of finding the categories in the paragraphs represents what we aim for this experiment with students.

**Note 1:** We decided to use spreadsheets especially because they save time (i.e. the students will focus only in the analysis, instead of spending time with steps such as the preprocessing of the files), and because the spreadsheets are easy to follow.

**Note 2:** We already downloaded the documentation files for the projects we decided to use, and we already (manually) created the spreadsheets for each one of them. The spreadsheets are available in [this folder](https://github.com/fronchetti/ICSE-2019/tree/master/march-report/spreadsheets).

## Step 2:
### Write a tutorial on how to (qualitatively) code the documentation files:

After the discussion on how to conduct the experiments, we wrote the tutorial on how students should do the documentation analysis. We wil maintain the tutorial updated on [this page](https://fronchetti.github.io/ICSE-2019/).

## Step 3:
### Conduct a pilot analysis

The  pilot analysis was a little bit different from what we expected. Igor Wiese, Steinmacher and Christoph were not available for the experiment. Gustavo proposed to participate in the pilot analysis with one of his students from UFPA. Felipe was the responsible for conducting the experiment. They read the tutorial, and tried to codify the README and CONTRIBUTING files for the [Flask](https://github.com/pallets/flask) project. After the experiment, they pointed out some observations and suggestions:

* Improve the definition for the contribution flow: "It is very difficult to understand the difference between the contribution flow and every information available in a documentation file".
* Discuss carefully with the students how the experiment should be conducted, giving the following advices:
  * During the experiment, the students do not need to mark a category for all the paragraphs.
  * If a paragraph is unclear and/or does not contain relevant information, the student can skip it.
  * Two or more paragraphs may be related to a same subject. 
  * The student can mark as many categories as he/she judge necessary. The analysis should be based on the participant's interpretation of the paragraphs. 
* Remove special characters that makes the paragraphs difficult to understand (e.g. Characters used in the Markdown syntax) before the final experiment. 
* The time spent to code the documentation files was smaller than expected. We can give more than a project per student.

The codifications made by Gustavo and his student are available in [this folder](https://github.com/fronchetti/ICSE-2019/tree/master/march-report/analysis/pilot). The suggestions made by them were already applied in the [students tutorial](https://fronchetti.github.io/ICSE-2019/).

Felipe, Gustavo and his student also discussed some possible limitations for the experiment, such as: the time available to apply the experiment, the students understanding on concepts used in software engineering, the students difficulties when reading texts in English, and the limited physical space we will have to apply the experiment (especially, the number of computers available).

## STEP 4 (Part I: UFPA)
### Conduct the analysis with undergraduate students
After the pilot analysis, we decided to do the first experiment with students. The experiment was applied on March, 29, at the Federal University of Pará, and was guided by the professor and co-author of this paper, Gustavo Pinto. Gustavo prepared a presentation about FLOSS, which included the definitions for the categories we would like to identify in projects documentation. More than 40 students participated in the experiment. The analysis made by the students is available in [this folder](https://github.com/fronchetti/ICSE-2019/tree/master/march-report/analysis/ufpa) (We have removed any student-related information before publishing it online).

After the experiment, Gustavo reported some observations:
* Some students (~3) did not have enough experience with technologies such as GitHub. Maybe we could think in something to help them understand the basics.
* Some students did not have enough knowledge on the English language (which was expected).
* Some students tried to follow the links on the documentation files. It is a good idea to tell them to only use what is available in the documentation files (Otherwise, they will spend a lot of time looking for information in related websites).
* One student asked if reporting a bug is part of the contribution flow (Gustavo said yes, but we should discuss it carefully).

The observations made by Gustavo were discussed by the team, and the [students tutorial](https://fronchetti.github.io/ICSE-2019/) was updated.

### A few observations after the experiment:
#### Number of occurrences per category:<br>
 FM – Find a mentor: 22<br>
 SC – Submit the changes: 33<br>
 CT – Choose a task: 40<br>
 TC – Talk to the community: 84<br>
 CF – Contribution flow: 107<br>
 BW – Build local workspace: 141<br>
 DC – Deal with the code: 162
 
 ![](https://github.com/fronchetti/ICSE-2019/blob/master/march-report/categories-ufpa.png)

### Cool things that happened this month:
* Professor Daniel from the University of São Paulo (USP) invited us to apply our experiments in his class about Open Source development. Now we can ask a group of undergraduate students to help us with the documentation analysis.
* Gustavo will also be giving lessons about Open Source development to students of the Federal University of Pará (UFPA). More potential students to help us in the analysis! 
