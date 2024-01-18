# An extended demographically informed Toontown Rewritten NLP analysis

## Description and background
The goal of this project was to expand upon previous demographically informed Toontown Rewritten NLP research in hopes of obtaining more extensive and reportable results. Previous work in this area has not been very informative and it has been hypothesized that the primary reason for this is a lack of data. In order to carry out a more in-depth analysis, a second demographic corpus containing 6000 chat messages and pieces of demographic information related to the speaker of each message was compiled between August and October 2023 and merged with the original demograhic corpus compiled for the preliminary analysis in this area published in July 2023.

## Methodology, implementation
This analysis was carried out in Python. NumPy and pandas were used for data cleaning and manipulation, VADER and TextBlob were used to calculate message sentiment and subjectivity scores, and SciPy was utilized for statistical testing. It had been my intention to use the spaCy entity recognizer on this data, but there were catastrophic issues with its output. 

## Results and observations
Once again, few interesting or notable results were yielded by this analysis, even with a more diverse set of linguistic metrics having been examined. Findings displayed a remarkable level of similarity to those of prior research with only a small handful of novel significant results predominantly related to differences in message length and sentiment between genders. It is possible that a lack of data remains a limitation but future research is not currently planned in this area.

## Repository contents
This repository contains a paper reporting on the analysis and its results, both demographic corpora used to create the extended demographic corpus, and a bibtex file. 

## Status
Due to major changes in the game's gender system, the difficulty of collecting large amounts of data by hand, and extremely similar results yielded across multiple studies, this specific research area has been indefinitely put on hold. There are some tentative plans to potentially use data from the demographic corpora in a machine learning context, but much strategization is necessary regarding this research direction, as well as plans to create something interactive and visual, such as a dashboard.
