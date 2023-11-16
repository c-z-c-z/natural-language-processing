# LDA and NMF analysis on Toontown Rewritten corpus

## Description and background
In this project, I performed topic modeling on my preexisting corpus of 4000 Toontown Rewritten chat messages through latent Dirichlet allocation and non-negative matrix factorization. The goal was to cluster the messages of the corpus into ten topics in order to see if any meaningful groupings could be achieved and if any of the most frequent words for a topic output by either algorithm would correspond to a cohesive subject. Can substantive topic modeling be carried out on a corpus of this size with such a diverse range of short messages?

## What are LDA and NMF?
Latent Dirichlet allocation represents documents as groups of topics, each of which output words at different probabilities. LDA assumes that documents are produced by choosing a topic mixture for the document and generating each word in the document according to this topic distribution, then backtracks from the documents to find this topic mixture which is likely to have generated the document at hand. Each word is randomly assigned to one of the user-determined number of topics at random, and as the algorithm iterates over each word, topics are reassigned based on the calculated probability that a certain topic generated the word until a steady state is reached. 

Non-negative matrix factorization performs dimensionality reduction and clustering. A document term matrix is created using TF-IDF, and the algorithm attempts to find a k-dimension approximation in terms of two non-negative factors. Each object, or column of the matrix, is approximated by a linear combination of k reduced dimensions, or basis vectors, each of which can be interpreted as a cluster. We get a measurement of reconstruction error between the original matrix and the approximation and attempt to optimize and refine the approximation to minimize error. Coefficients are updated until the approximation makes sense. 

## Implementation
This analysis was performed in Python using scikit-learn and VADER. matplotlib and seaborn were utilized for visualization and pandas and numpy for data manipulation.

## Results and observations
Both methods were unsuccessful. Neither LDA nor NMF yielded meaningful results or coherent topics. There was also almost no similarity across the two methods regarding to which topic a message was assigned. I hypothesize that this stems from multiple factors: the relatively small size of the corpus, the shortness of the messages, their online chat context origin, and the fact that a sizable proportion of the messages are very informal and do not have an easily discernible or specific subject.

## Repository contents
This repository contains two code files and the message corpus. I have included my Jupyter Notebook to handily display visualizations. 

## Status
Initial analysis is complete, but the inconclusiveness of these results is frustrating and fascinating. I aim to compile a much larger message corpus and attempt once again to carry out topic modeling. As of performing a couple of tiny updates to this text in November 2023, this much larger message corpus has successfully been compiled, and I very much intend to perform topic modeling once again in the near future.
