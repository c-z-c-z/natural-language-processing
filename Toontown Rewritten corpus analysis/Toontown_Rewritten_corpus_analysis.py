#!/usr/bin/env python
# coding: utf-8

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import wordcloud
import nltk
import string
import textblob

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')


#Reading in the data
corpus = pd.read_csv('corpus.csv', header=None, names=['text'])

#Splitting message into individual tokens
corpus['tokens'] = corpus['text'].str.split(' ')

#Appending every individual word in the corpus to a list
words = []
for i in range(corpus.shape[0]):
    message_length = len(corpus['tokens'].iloc[i])
    for j in range(message_length):
        words.append(corpus['tokens'].iloc[i][j])

#Removing capitalization and punctuation from the list of all individual words
cleanwords = []
for word in words:
    word = word.lower()
    word = word.translate(str.maketrans('', '', string.punctuation))
    cleanwords.append(word)

#Message length in tokens
corpus['length'] = corpus['tokens'].apply(len)

#Message length in characters
corpus['charlength'] = corpus['text'].apply(len)


#Function to get textblob subjectivity and polarity for each message
def get_subjectivity(text):
    return textblob.TextBlob(text).sentiment.subjectivity

def get_polarity(text):
    return textblob.TextBlob(text).sentiment.polarity

corpus['subjectivity'] = corpus['text'].apply(get_subjectivity)
corpus['polarity'] = corpus['text'].apply(get_polarity)

#Using polarity value to determine message sentiment judgment

def get_analysis(score):
    if score < 0:
        return 'negative'
    elif score == 0:
        return 'neutral'
    else:
        return 'positive'

corpus['analysis'] = corpus['polarity'].apply(get_analysis)

#Positive, negative, neutral messages
pos = corpus[corpus['analysis'] == 'positive']
neg = corpus[corpus['analysis'] == 'negative']
neutral = corpus[corpus['analysis'] == 'neutral']

#Percentages for each category
posp = round((pos.shape[0]/4000)*100, 1)
negp = round((neg.shape[0]/4000)*100, 1)
neutralp = round((neutral.shape[0]/4000)*100, 1)

percentages = f"positive: {posp}%, negative: {negp}%, neutral: {neutralp}%"
print(percentages)
#positive: 20.5%, negative: 9.8%, neutral: 69.7%

pos.describe()
neg.describe()
neutral.describe()

#Generating a word cloud which can be found in the visualizations folder
cleanwords2 = ' '.join([word for word in cleanwords])
wc = wordcloud.WordCloud(width=1000, height=1000, random_state=24, 
              max_font_size=125, background_color='white').generate(cleanwords2)

#Using vaderSentiment

#Dropping the values we just generated with textblob
corpus_vader = corpus.drop(columns=['subjectivity', 'polarity', 'analysis'])

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

corpus_vader['scores'] = corpus_vader['text'].apply(analyzer.polarity_scores)

#Determining sentiment judgment from analyzer's scores

def get_analysis(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

#Compound score indicates overall polarity of sentiment as judged by analyzer
#We must apply the function above to this compound score

compounds = []
for i in range(4000):
    compounds.append(corpus_vader['scores'].iloc[i]['compound'])
corpus_vader['compounds'] = compounds

corpus_vader['analysis'] = corpus_vader['compounds'].apply(get_analysis)

#Positive, negative, neutral messages
vpos = corpus_vader[corpus_vader['analysis'] == 'positive']
vneg = corpus_vader[corpus_vader['analysis'] == 'negative']
vneu = corpus_vader[corpus_vader['analysis'] == 'neutral']

#Percentages for each category
vp = round((vpos.shape[0]/4000)*100, 1)
vn = round((vneg.shape[0]/4000)*100, 1)
vne = round((vneu.shape[0]/4000)*100, 1)

vpercentages = f"positive: {vp}%, negative: {vn}%, neutral: {vne}%"
print(vpercentages)
#positive: 29.1%, negative: 13.2%, neutral: 57.7%

vpos.describe()
vneg.describe()
vneu.describe()

#vS can better interpret online speech nuances
#Numerical/statistical features of each group remain largely the same between analyzers

#Finding instances where the two analyzers differ in their assessment

corpus2 = corpus.copy()
corpus_vader2 = corpus_vader.copy()

corpus2['analyzer'] = 'textblob'
corpus_vader2['analyzer'] = 'vaderSentiment'

corpus2.drop(columns=['polarity'], inplace=True)
corpus_vader2.drop(columns=['scores'], inplace=True)

#Indicating which analyzer returned the sentiment judgment in the column
corpus2.rename(columns={'analysis': 'textblob'}, inplace=True)
corpus_vader2.rename(columns={'analysis': 'vaderSentiment'}, inplace=True)

#Finding indices where judgments differ between the 2 analyzers
#Iterating over textblob, VADER judgments simultaneously, noting where they aren't the same

indices = []
for i in range(4000):
    if corpus2.iloc[i]['textblob'] != corpus_vader2.iloc[i]['vaderSentiment']:
        indices.append(i)

#Dropping the rows which do NOT differ
corpus2.drop(axis=0, index=[r for r in np.arange(0,4000) if r not in indices], inplace=True)
corpus_vader2.drop(axis=0, index=[r for r in np.arange(0,4000) if r not in indices], inplace=True)

#Adding to the textblob df an indication of which sentiment VADER chose for each message
corpus2['vaderSentiment'] = corpus_vader2['vaderSentiment']

#Looking at only the messages themselves
analyses = corpus2.drop(columns=['subjectivity', 'length', 'charlength', 'analyzer'])

analyses.head(20)

#Looking at most frequent tokens

#you and u are two of the most frequently occurring tokens
#I'm going to look at them from two different perspectives

#Treating you and u as different tokens
corpus_df = pd.DataFrame(cleanwords[0].split(), columns=['token'])
counts = corpus_df.value_counts().rename_axis('token').reset_index(name='counts')

#Treating you and u as the same token
corpus_df2 = corpus_df.copy()
corpus_df2.replace({'u':'you'}, inplace=True)
counts2 = corpus_df2.value_counts().rename_axis('token').reset_index(name='counts')

#Generated many plots and visuals which are located in the visualization folder 

