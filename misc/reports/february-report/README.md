# February Report

In this first report, we describe the findings we have made in a preliminary analysis of documentation files for 10 Open Source Software (OSS) projects. We decided in the last weeks, to make a study on how information in documentation files is arranged to improve the interest and understanding of newcomers while they are attempting to contribute with an OSS project. Here is a draft of our motivations:

> Documentation are not well defined in OSS projects (sometimes spread, sometimes outdated, sometimes nonexistent). Newcomers need documentation to understand how a project works and to start with the contribution process. Why not create a classifier that identifies sections in documentation files that might be of interest to newcomers?

## Approach:
Initially, we would like to understand if there is significant information for newcomers in OSS repositories (considering README and CONTRIBUTING files), then we would like to comprehend how this information is arranged and which categories we can create from these documents. 

**Dataset:**
Using 450 projects from a [previous study](https://github.com/fronchetti/OSS-2019), we made a summary of the total of newcomers per project:

| Min  |  1st Qu.  | Median | 3rd Qu. | Max.
|------|-----------|--------|---------|---
| 0.0  |  43.0 | 100.0 | 293.8 | 8431.0 

We selected a total of 10 projects to be analyzed. To better represent the diversity of projects in OSS communities, we randomly selected 5 projects that attract more (or equally) newcomers as the median of projects in the dataset, and 5 projects that attract fewer newcomers than the median. The list of projects is as follows:

* [Cocoa](https://github.com/realm/realm-cocoa) (Objective-C)<br>
* [Consul](https://github.com/hashicorp/consul) (Go)<br>
* [Akka](https://github.com/akka/akka) (Scala)<br>
* [MySQLTuner](https://github.com/major/MySQLTuner-perl) (Pearl)<br>
* [Flask](https://github.com/pallets/flask) (Python)<br>
* [Rebar](https://github.com/basho/rebar) (Erlang)<br>
* [Torch7](https://github.com/torch/torch7) (C)<br>
* [PHP-SRC](https://github.com/php/php-src) (C)<br>
* [AFNetworking](https://github.com/AFNetworking/AFNetworking) (Objective-C) <br>
* [Scala](https://github.com/scala/scala) (Scala) <br>

### Analyzing the files:

To better comprehend what kind of information we can find in README and CONTRIBUTING files, we classified the documentation files of the 10 OSS projects considering the [Barriers Model](https://www.ime.usp.br/~cpg/teses/Tese-IgorFabioSteinmacher.pdf) proposed by Igor Steinmacher. After a few classification attempts, we found seven types of information in documentation files that might contribute with the onboarding of newcomers:
* How to start with the project (Barrier(s): Finding a task to start with)
* What is the contribution flow (Barrier(s): Newcomers don't know the contribution flow)
* How to submit a contribution (Barrier(s): Lack of information on how to send a contribution)
* How to find a mentor (Barrier(s): Finding a mentor, Newcomers need to contact a real person)
* Where is the documentation (Barriers(s): Lack of documentation, Spread documentation)
* How to build workspace locally (Barrier(s): Building workspace locally)
* What are the projects process and pratices (Barriers(s): Lack of knowledge in project process and practices, Lack of code standards)

All the files were manually classified, and are now available in this folder:

* [GitHub](https://github.com/fronchetti/IME-USP/tree/master/february-report/spreadsheets)

Here is a quick summary of the occurences per category:
![](https://github.com/fronchetti/IME-USP/blob/master/february-report/barriers.png)

### Future work:
Discuss the projects that we have analyzed and the categories we have found in a future meeting. <br>
If we come to an agreement, we can look at other projects.

### Cool things that happened this month:
**Felipe:** I am participating of the [FLUSP](https://flusp.ime.usp.br/
) (FLOSS at USP). During a Telegram discussion, a student reported an interesting thing about her experience trying to contribute to the Linux Kernel:

![](https://github.com/fronchetti/IME-USP/blob/master/february-report/newcomer-report.jpg)

".. Following a tutorial is simple, since the information is already organized, but knowing how to find those steps is the most difficult process."

What a great motivation for this work :-)!

### Meetings:
* [19/02/2019](https://github.com/fronchetti/IME-USP/tree/master/meetings/meeting-19-02-2019) 
