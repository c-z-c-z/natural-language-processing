#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import string

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

corpus = pd.read_csv('corpus.csv', header=None, names=['message'])

#Creating an extensive list of stop words to remove prior to analysis
from sklearn.feature_extraction import _stop_words
import nltk
from nltk.corpus import stopwords

nopuncwords = []
for word in stopwords.words('english'):
    nopunc = word.translate(str.maketrans('', '', string.punctuation))
    nopuncwords.append(word)
    if word != nopunc:
        nopuncwords.append(word.translate(str.maketrans('', '', string.punctuation)))
    
sklearnwords = []
for word in _stop_words.ENGLISH_STOP_WORDS:
    sklearnwords.append(word)

#Compiling all words in the corpus into a list
#Splitting messages into tokens, cleaning words by rendering lowercase & removing punctuation

corpus['tokens'] = corpus['message'].str.split(' ')

words = []
for i in range(corpus.shape[0]):
    message_length = len(corpus['tokens'].iloc[i])
    for j in range(message_length):
        words.append(corpus['tokens'].iloc[i][j])

cleanwords = []
for word in words:
    word = word.lower()
    word = word.translate(str.maketrans('', '', string.punctuation))
    cleanwords.append(word)

#Creating a data frame to display counts for each unique token
word_df = pd.DataFrame(cleanwords[0].split(), columns=['token'])
freq = word_df.value_counts().rename_axis('token').reset_index(name='counts')

#Adding additional personalized stop words based on these counts

stopwords2 = 'i, the, you, a, to, u, my, it, is, me, do, im, and, for, are, like, so, that, in, have, what, on, of, we, get, \
hi, be, but, its, he, was, can, ok, got, how, they, why, with, this, ur, thats, she, your, at, if, will, did, too, more, oh, when, \
am, them, or, some, out, hey, where, whats, lets, then, from, there, still, her, him, us, were, gonna, much, been, about, could, \
any, youre, would, shes, hes, should, would, could, had, his, our, off, tho, as, r, does, doing, did, ive, than, okay, rn, has, an, 
mine, us, havent, ago, yo, isnt, being, while, by, bc, hello, though, done, whos, having, cuz, went'
stopwords_list = stopwords2.split(',')

trimmed_sw = []
for word in stopwords_list:
    trimmed_sw.append(word.strip())
stopwords = nopuncwords + sklearnwords + trimmed_sw

#LDA analysis
#LDA depends on word count probabilities, so we use CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words=stopwords)
dtm = cv.fit_transform(corpus['message'])

#Attempting to group into 10 topics
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=10, random_state=114)
lda.fit(dtm)

#Top 20 words for each topic
for i, topic in enumerate(lda.components_):
    print(f"Top 20 words for topic #{i+1}:")
    print([cv.get_feature_names()[index] for index in topic.argsort()[-20:]])
    print('\n')

#Not particularly informative
#No cohesive topics to be found - very interesting mix of quite specific words and broad, general words

topic_results = lda.transform(dtm)
corpus['LDA_topic'] = (topic_results.argmax(axis=1)+1)

#Calculating some metrics, length of messages in words and characters
corpus['wordlength'] = corpus['tokens'].apply(len)
corpus['charlength'] = corpus['message'].apply(len)

#Calculating polarities
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
corpus['scores'] = corpus['message'].apply(analyzer.polarity_scores)

#Creating a simple function to sort each message based on compound polarity score
def get_analysis(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

#Applying this function to each message's calculated compound polarity score
compounds = []
for i in range(len(corpus)):
    compounds.append(corpus['scores'].iloc[i]['compound'])
corpus['compounds'] = compounds
corpus['polarity'] = corpus['compounds'].apply(get_analysis)
corpus.drop(columns=['compounds', 'scores'], inplace=True)

#Collecting some metrics to perform visualization
#Preparing to compare means and standard deviations for message lengths in words & characters
wordlen_mean = dict.fromkeys(range(1,11))
wordlen_std = dict.fromkeys(range(1,11))
charlen_mean = dict.fromkeys(range(1,11))
charlen_std = dict.fromkeys(range(1,11))

#Collecting mean and standard deviation for message lengths for each topic
for i in range(1,11):
    wordlen_mean[i] = corpus[corpus['LDA_topic'] == i].describe()['wordlength']['mean']
wordmeans = list(wordlen_mean.values())
    
for i in range(1,11):
    wordlen_std[i] = corpus[corpus['LDA_topic'] == i].describe()['wordlength']['std']
wordstds = list(wordlen_std.values())

for i in range(1,11):
    charlen_mean[i] = corpus[corpus['LDA_topic'] == i].describe()['charlength']['mean']
charmeans = list(charlen_mean.values())
    
for i in range(1,11):
    charlen_std[i] = corpus[corpus['LDA_topic'] == i].describe()['charlength']['std']
charstds = list(charlen_std.values())

import matplotlib.pyplot as plt
import seaborn as sns

metrics = pd.DataFrame({'topic':np.arange(1,11),
                               'word_len_mean': wordmeans,
                              'word_len_std': wordstds,
                              'char_len_mean': charmeans,
                              'char_len_std': charstds},
                             index=np.arange(1,11))

topics = range(1,11)
means = metrics['word_len_mean'].values
sd = metrics['word_len_std'].values
char = metrics['char_len_mean'].values
charsd = metrics['char_len_std'].values

#Jupyter Notebook in the repository displays these visualizations
#fig, ax = plt.subplots(figsize=(12,6))
#ax.bar(range(1,11), means, color='Purple', ecolor='Black', yerr=sd, alpha=0.5, width=0.8, capsize=5)
#ax.set_xticks(topics)
#ax.set_yticks(np.arange(0,9))
#ax.set_title('Mean message length in tokens: topic assignment with LDA')
#ax.set_xlabel('Topic')
#ax.set_ylabel('Length')
#ax.yaxis.grid(True);

#fig, ax = plt.subplots(figsize=(12,6))
#ax.bar(range(1,11), char, color='Orange', ecolor='Black', yerr=charsd, alpha=0.75, width=0.8, capsize=5)
#ax.set_xticks(topics)
#ax.set_yticks(np.arange(0,41,5))
#ax.set_title('Mean message length in characters: topic assignment with LDA')
#ax.set_xlabel('Topic')
#ax.set_ylabel('Length')
#ax.yaxis.grid(True);

#Creating a data frame with exclusively sentiment-related info
polarities = corpus.drop(columns=['message', 'tokens', 'wordlength', 'charlength'])

#Custom colours
palette1 = sns.color_palette(['#68966b', '#7b9ded', '#c46e6e'])

plt.figure(figsize=(12,6))
sns.countplot(data=polarities, x='LDA_topic', hue='polarity', 
              hue_order=['positive', 'neutral', 'negative'], palette=palette1)
plt.title('Polarity distribution by topic: assignment with LDA')
plt.xlabel('Topic');

#Nothing jumps out except for the much higher amount of messages assigned to topic 1
#Given the apparent randomness and lack of cohesion of the topics, I don't know why we observe this distribution

#NMF analysis
#We use tfidf vectorization because we use coefficients for NMF

#Trying to strike a balance between terms which are very unique and those which are very common
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=4, stop_words=stopwords)
dtm_nmf = tfidf.fit_transform(corpus['message'])

from sklearn.decomposition import NMF
nmf = NMF(n_components=10, random_state=114)
nmf.fit(dtm_nmf)

for index, topic in enumerate(nmf.components_):
    print(f"Top 20 words for topic {index+1}:")
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-20:]])
    print('\n')

topic = nmf.transform(dtm_nmf)
corpus['NMF_topic'] = (topic.argmax(axis=1)+1)

#Checking out metrics again for NMF topics
wordlen_mean2 = dict.fromkeys(range(1,11))
wordlen_std2 = dict.fromkeys(range(1,11))
charlen_mean2 = dict.fromkeys(range(1,11))
charlen_std2 = dict.fromkeys(range(1,11))

for i in range(1,11):
    wordlen_mean2[i] = corpus[corpus['NMF_topic'] == i].describe()['wordlength']['mean']
wordmeans2 = list(wordlen_mean2.values())
    
for i in range(1,11):
    wordlen_std2[i] = corpus[corpus['NMF_topic'] == i].describe()['wordlength']['std']
wordstds2 = list(wordlen_std2.values())

for i in range(1,11):
    charlen_mean2[i] = corpus[corpus['NMF_topic'] == i].describe()['charlength']['mean']
charmeans2 = list(charlen_mean2.values())
    
for i in range(1,11):
    charlen_std2[i] = corpus[corpus['NMF_topic'] == i].describe()['charlength']['std']
charstds2 = list(charlen_std2.values())

metrics2 = pd.DataFrame({'topic':np.arange(1,11),
                               'word_len_mean': wordmeans2,
                              'word_len_std': wordstds2,
                              'char_len_mean': charmeans2,
                              'char_len_std': charstds2},
                             index=np.arange(1,11))

topics = range(1,11)
means2 = metrics['word_len_mean'].values
sd2 = metrics['word_len_std'].values
char2 = metrics['char_len_mean'].values
charsd2 = metrics['char_len_std'].values

#Jupyter Notebook in the repository displays these visualizations
#fig, ax = plt.subplots(figsize=(12,6))
#ax.bar(range(1,11), means2, color='Green', ecolor='Black', yerr=sd2, alpha=0.5, width=0.8, capsize=5)
#ax.set_xticks(topics)
#ax.set_yticks(np.arange(0,9))
#ax.set_title('Mean message length in tokens: topic assignment with NMF')
#ax.set_xlabel('Topic')
#ax.set_ylabel('Length')
#ax.yaxis.grid(True);

#fig, ax = plt.subplots(figsize=(12,6))
#ax.bar(range(1,11), char2, color='Navy', ecolor='Black', yerr=charsd2, alpha=0.75, width=0.8, capsize=5)
#ax.set_xticks(topics)
#ax.set_yticks(np.arange(0,41,5))
#ax.set_title('Mean message length in characters: topic assignment with NMF')
#ax.set_xlabel('Topic')
#ax.set_ylabel('Length')
#ax.yaxis.grid(True);

#Data frame for comparing sentiment info across NMF-assigned topics
polarities2 = corpus.drop(columns=['message', 'tokens', 'wordlength', 'charlength', 'LDA_topic'])

plt.figure(figsize=(12,6))
sns.countplot(data=polarities2, x='NMF_topic', hue='polarity', 
              hue_order=['positive', 'neutral', 'negative'], palette=palette1)
plt.title('Polarity distribution by topic: assignment with NMF')
plt.xlabel('Topic');

#Topic 1 has been assigned the very large majority of messages and some topics have nearly zero!

#How many messages ended up in each NMF topic?
topic_counts = corpus['NMF_topic'].value_counts()
for i in range(1,11):
    print(f"Topic {i}:", topic_counts[i], "messages")

#Why is this distribution the way that it is?
#Once again, based on the top 20 words for each topic, no topic has cohesive, specific subject matter
#There is also no similarity between LDA and NMF in terms of the top 20 words for each of the 10 topics


# In[ ]:




