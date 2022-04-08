
# Toontown Rewritten corpus analysis

## Description and background
The aim of this project was to explore a corpus of in-game chat messages from the MMORPG Toontown Rewritten primarily through calculating and examining metrics related to sentiment, namely polarity and subjectivity, as well as metrics such as message length and token frequencies. 4000 messages were compiled by hand throughout March 2022.   

## Implementation and libraries
This analysis was undertaken in Python primarily with the help of textblob and vaderSentiment. matplotlib, seaborn, and wordcloud were used for visualization, and pandas and numpy were used for data manipulation.

## Results and observations
Messages deemed neutral comprised the majority of the corpus for both analyzers (textblob and vaderSentiment). However, textblob judged 69.7% of messages in the corpus to be neutral while vaderSentiment returned 57.7%, which is likely due to vaderSentiment being more specifically attuned to online communication contexts and thus able to tease out meaning from messages textblob could not actually understand and therefore deemed neutral. This analyzer was able to correctly interpret many acronyms and abbreviations, such as 'ty', 'yw', 'np', 'jk', 'ez', 'ftw', and 'ily', and judge their sentiment appropriately.

vaderSentiment also returned more nuanced results. It often appeared that textblob would notice that the message contained a word such as "more", "new", "high", or "like" and would assign the message positive sentiment based on this, with the same thing also occurring in the case of negative sentiment, while vaderSentiment was able to more correctly extract the true meaning of the message. For example, textblob judged the message "i never have trouble finding them" to be negative while vaderSentiment correctly identified its positive sentiment, and textblob called "i see what u mean" negative while vaderSentiment was able to parse it as neutral.

For both analyzers, mean values of subjectivity, message length in tokens, message length in characters, and standard deviations for these measures for all three polarity categories were nearly identical, with only neutral messages appearing to possibly tend to be a few characters shorter in the case of textblob. 

## Repository contents
This repository consists of two code files, the message corpus, a .txt file of all individual words in the corpus, and various visualizations created with matplotlib and seaborn.

## Status
Initial analysis is complete, although there is much more to be explored. I plan to go into more detail regarding distribution of certain tokens and to complete a more comprehensive research style writeup related to this.
