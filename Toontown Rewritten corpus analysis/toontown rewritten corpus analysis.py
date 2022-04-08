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
os.chdir('/Users/ethancz/Library/Python/3.8/lib/python/site-packages')
import textblob

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

corpus = pd.read_csv('ttr corpus.csv', header=None, names=['text'])

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

#Functions to get subjectivity and polarity for each message
def get_subjectivity(text):
    return textblob.TextBlob(text).sentiment.subjectivity

def get_polarity(text):
    return textblob.TextBlob(text).sentiment.polarity

corpus['subjectivity'] = corpus['text'].apply(get_subjectivity)
corpus['polarity'] = corpus['text'].apply(get_polarity)

#Using polarity value to determine if message has been deemed positive, negative, neutral
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
print(posp, negp, neutralp)

pos.describe()
neg.describe()
neutral.describe()

#Generating a word cloud
cleanwords2 = ' '.join([word for word in cleanwords])
wc = wordcloud.WordCloud(width=1000, height=1000, random_state=24, 
              max_font_size=125, background_color='white').generate(cleanwords2)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')


#Using vaderSentiment

corpus_vader = corpus.drop(columns=['subjectivity', 'polarity', 'analysis'])

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

corpus_vader['scores'] = corpus_vader['text'].apply(analyzer.polarity_scores)

#Determining if message has been deemed positive, negative, neutral
def get_analysis(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


#compound score indicates overall polarity of sentiment as judged by analyzer
compounds = []
for i in range(4000):
    compounds.append(corpus_vader['scores'].iloc[i]['compound'])
corpus_vader['compounds'] = compounds
corpus_vader['analysis'] = corpus_vader['compounds'].apply(get_analysis)
corpus_vader.drop(columns=['compounds'], inplace=True)

#Positive, negative, neutral messages
vpos = corpus_vader[corpus_vader['analysis'] == 'positive']
vneg = corpus_vader[corpus_vader['analysis'] == 'negative']
vneu = corpus_vader[corpus_vader['analysis'] == 'neutral']

#Percentages for each category
vp = round((vpos.shape[0]/4000)*100, 1)
vn = round((vneg.shape[0]/4000)*100, 1)
vne = round((vneu.shape[0]/4000)*100, 1)
print(vp, vn, vne)


vpos.describe()
vneg.describe()
vneu.describe()


#vS can better interpret online speech nuances, but the numerical features of each group
#remain the same as those of the groups as determined by textblob

#Finding instances where the two analyzers differ in their assessment

corpus2 = corpus.copy()
corpus_vader2 = corpus_vader.copy()
corpus2['analyzer'] = 'textblob'
corpus_vader2['analyzer'] = 'vaderSentiment'

corpus2.drop(columns=['polarity'], inplace=True)
corpus_vader2.drop(columns=['scores'], inplace=True)

corpus2.rename(columns={'analysis': 'textblob'}, inplace=True)
corpus_vader2.rename(columns={'analysis': 'vaderSentiment'}, inplace=True)


#Finding indices where polarity judgments differ between the 2 analyzers
indices = []
for i in range(4000):
    if corpus2.iloc[i]['textblob'] != corpus_vader2.iloc[i]['vaderSentiment']:
        indices.append(i)

#Dropping rows which do not differ
corpus2.drop(axis=0, index=[r for r in np.arange(0,4000) if r not in indices], inplace=True)
corpus_vader2.drop(axis=0, index=[r for r in np.arange(0,4000) if r not in indices], inplace=True)

corpus2['vaderSentiment'] = corpus_vader2['vaderSentiment']

corpus2.describe()
corpus_vader.describe()


#Looking at only the messages themselves
analyses = corpus2.drop(columns=['subjectivity', 'length', 'charlength', 'analyzer'])

analyses


#Treating you and u as different tokens
corpus_df = pd.DataFrame(cleanwords[0].split(), columns=['token'])
counts = corpus_df.value_counts().rename_axis('token').reset_index(name='counts')

#Treating you and u as the same token
corpus_df2 = corpus_df.copy()
corpus_df2.replace({'u':'you'}, inplace=True)
counts2 = corpus_df2.value_counts().rename_axis('token').reset_index(name='counts')

#Some plotting...

#Looking at the characteristics of these two tokens a bit more closely

u_indices = []
you_indices = []

for i in range(4000):
    if 'you' in corpus['tokens'].iloc[i]:
        you_indices.append(i)
    elif 'u' in corpus['tokens'].iloc[i]:
        u_indices.append(i)
    else:
        continue

corpus3 = corpus.copy()
you = corpus3.drop(axis=0, index=[i for i in np.arange(4000) if i not in you_indices])
u = corpus3.drop(axis=0, index=[i for i in np.arange(4000) if i not in u_indices])

you.describe()
u.describe()

