
# Dota 2 corpus analysis

## Description and background
This project explored and visualized a large corpus of in-game chat messages from the game Dota 2 through measures of token frequency, polarity, and message length. 

## Implementation and libraries
This analysis was undertaken in Python primarily with the help of VADER. VADER was opted for over textblob because it is specifically attuned to social media contexts. matplotlib, seaborn, and wordcloud were used for visualization, and pandas and numpy were used for data manipulation. Read more about VADER here: https://github.com/cjhutto/vaderSentiment

## Dataset
The dataset is far too large to be added to the repository. It can be downloaded from https://www.kaggle.com/datasets/devinanzelmo/dota-2-matches. Many thanks to Devin Anzelmo for making this available!

## Repository contents
This repository consists of two code files, two .txt files of individual words in the corpus (with and without non-Latin characters removed), and various visualizations created with matplotlib and seaborn.

## Status
Initial analysis is complete. I plan to go into more detail in the near future regarding examination of metrics for each polarity category to attempt to determine if they differ systematically in any way.
