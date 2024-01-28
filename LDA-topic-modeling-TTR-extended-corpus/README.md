# LDA topic modeling with gensim on extended TTR demographic corpus

## Description and background
Topic modeling was performed via latent Dirichlet allocation on a corpus consisting of 10,000 chat messages from the game Toontown Rewritten and various pieces of demographic information about the speaker of each message. LDA was first performed with scikit-learn in May 2022 on the original non-demographic corpus which consisted of 4,000 TTR chat messages. Messages in the extended demographic corpus, completed in October 2023 for an expanded demographic NLP analysis published earlier in January 2024, were clustered into ten topics which were subsequently examined. Gensim was selected over scikit-learn due to its optimization for larger text corpora and various advanced functionalities.

## What is LDA?
Latent Dirichlet allocation is a probabilistic model which assumes that a topic is a mixture of an underlying set of word probabilities and that a document is a mixture of a set of topic probabilities. Given a number of documents, words, and number of topics, LDA models analyze word co-occurrence patterns and probabilistically assign words to topics and documents to distributions of topics.

## Implementation
This project was implemented in Python using spaCy and gensim. matplotlib and seaborn were utilized for visualization.

## Results and observations
Coherent topics were not uncovered by the model. Multiple characteristics of the chat messages at hand presumably contribute to this, namely their short length, online origin, and informality. There were a handful of somewhat interesting observations. Monkeys have a noticeably higher amount of messages concentrated in topic 4. The top word for topic 4 is "beans" and this is also the most frequent token uttered by monkeys as uncovered in the January 2024 demographic NLP study. Toons with higher Laff (60 and greater) are also noticeably concentrated in topic 4, but the frequency of "beans" as a token decreases as Laff increases, so this observation appears interestingly counterintuitive.

## Repository contents
The repository contains two code files and the demographic corpus. A Jupyter notebook has been included to display visualizations.

## Status
Further work is not currently planned in the Toontown sphere for various reasons (see January 2024 demographic NLP paper for more information; I do highly encourage anyone viewing this repository to take a glance at that paper in order to better understand the corpus and the information that it contains)
