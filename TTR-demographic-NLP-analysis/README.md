# A preliminary demographically informed Toontown Rewritten NLP analysis

## Description and background
This project's goal was to integrate demographic and NLP research related to Toontown Rewritten into a unified investigation. I have conducted previous work in both of these areas using corpora assembled from in-game information and believe that allowing them to inform each other will yield notable results and reveal important information. In order to carry out a preliminary analysis, a corpus consisting of 4000 chat messages and various pieces of demographic information about the speaker of each message was assembled between August 2022 and January 2023. 

## Methodology, implementation
This analysis was carried out primarily in R with the help of dplyr, stringr, VADER, as well as the tidyverse and tidytext. Visualizations were created using ggplot2. TextBlob was utilized in Python in order to calculate subjectivity. The PDF report was generated using R Markdown.

## Results and observations
Unfortunately, few results of interest were produced by this analysis and no broad conclusions can be drawn at this point. However, the lack of conclusive results presents a powerful implication that much further investigation is necessary here. Significantly larger amounts of data will be necessary to undertake this endeavour. I intend to resume data collection very soon in order to continue my research as well as expand into other areas with these corpora such as machine learning. As of November 2023, a second corpus of 6000 messages and corresponding demographic information has been completed and analysis of the two corpora as a single corpus will begin in the coming weeks.

## Repository contents
The repository contains an R Markdown file, a PDF report generated using this R Markdown, a BibTex file containing a list of references used in said report, and the demographically informed message corpus.

## Status
Initial investigation is complete and path is somewhat clear for further investigation as outlined in the final sections of the paper. As of November 2023, further data collection has been completed and another investigation will be launched very soon. This second investigation has been completed and published as of January 2024.
