#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud
import string

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

#Reading in the data
corpus = pd.read_csv('chat.csv')

#Dropping non-linguistic info
corpus.drop(columns=['match_id', 'unit'], inplace=True)

#Converting to strings and splitting messages into word tokens
corpus['key'] = corpus['key'].apply(str)
corpus['tokens'] = corpus['key'].str.split(' ')

#Calculating message length in both characters and words
corpus['length'] = corpus['tokens'].apply(len)
corpus['charlength'] = corpus['key'].apply(len)

#Appending each individul token to a list
words = []
for i in range(corpus.shape[0]):
    message_length = len(corpus['tokens'].iloc[i])
    for j in range(message_length):
        words.append(corpus['tokens'].iloc[i][j])
        
#Cleaning all tokens in list by removing punctuation & making lowercase
cleanwords = []
for word in words:
    word = word.lower()
    word = word.translate(str.maketrans('', '', string.punctuation))
    if word != '':
        cleanwords.append(word)

#Saving the tidy words to a text file
with open('words.txt', 'w') as f:
    for word in cleanwords:
        f.write(word)
        f.write('\n')
f.close()

#Examining the most frequently appearing words
word_df = pd.DataFrame(cleanwords, columns=['token'])
counts = word_df.value_counts().rename_axis('token').reset_index(name='counts')

#Checking out the top 10 most frequently appearing words
counts.head(10)
#gg, ez, lol, i, you, u, wp, report, is, a

#Generating a word cloud 
wc = wordcloud.WordCloud(width=1000, height=1000, random_state=24, 
              max_font_size=175, background_color='white',
                        collocations=False).generate(' '.join(cleanwords))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off');
#wc.to_file('dota word cloud.png');

#Dropping everything from the corpus which isn't written in the Latin alphabet
indices = []
for i in range(corpus.shape[0]):
    for char in corpus['key'].iloc[i].lower():
        if char not in "qwertyuiopasdfghjklzxcvbnm1234567890,=^@#$&_<>[]{}()!?.\/:;'+-*":
            indices.append(i)
            break
        break       
corpus_latin = corpus.copy()
corpus_latin.drop(axis=0, index=indices, inplace=True)

#Opting for VADER, not textblob, for polarity analysis
#VADER is specifically attuned to online contexts
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
corpus_latin['scores'] = corpus_latin['key'].apply(analyzer.polarity_scores)

#Creating function to sort message based on polarity score
def vader_analysis(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

#Collecting all compound scores (overall polarity judgment)
compounds = []
for i in range(corpus_latin.shape[0]):
    compounds.append(corpus_latin['scores'].iloc[i]['compound'])    
corpus_latin['compounds'] = compounds

#Applying function to compound score
corpus_latin['analysis'] = corpus_latin['compounds'].apply(vader_analysis)

#Positive, negative, neutral messages
pos = corpus_latin[corpus_latin['analysis'] == 'positive']
neg = corpus_latin[corpus_latin['analysis'] == 'negative']
neutral = corpus_latin[corpus_latin['analysis'] == 'neutral']

#Percentages for each category
posp = round((pos.shape[0]/corpus_latin.shape[0])*100, 1)
negp = round((neg.shape[0]/corpus_latin.shape[0])*100, 1)
neutralp = round((neutral.shape[0]/corpus_latin.shape[0])*100, 1)
print('Positive: {}%, negative: {}%, neutral: {}%'.format(posp, negp, neutralp))
#Positive: 35.9%, negative: 14.0%, neutral: 50.1%

#Positive, negative, neutral words for generation of word clouds
poswords = []
for i in range(pos.shape[0]):
    message_length = len(pos['key'].iloc[i].split())
    for j in range(message_length):
        poswords.append(pos['key'].iloc[i].split()[j].lower())
        
negwords = []
for i in range(neg.shape[0]):
    message_length = len(neg['key'].iloc[i].split())
    for j in range(message_length):
        negwords.append(neg['key'].iloc[i].split()[j].lower())
        
neutralwords = []
for i in range(neutral.shape[0]):
    message_length = len(neutral['key'].iloc[i].split())
    for j in range(message_length):
        neutralwords.append(neutral['key'].iloc[i].split()[j].lower())

#Word clouds for each sentiment judgment
wc = wordcloud.WordCloud(width=1000, height=1000, random_state=24, 
              max_font_size=250, background_color='white',
                        collocations=False).generate(' '.join(poswords))
plt.axis('off');
#wc.to_file('positive_cloud.png');

wc2 = wordcloud.WordCloud(width=1000, height=1000, random_state=24, 
              max_font_size=225, background_color='white',
                        collocations=False).generate(' '.join(negwords))
plt.axis('off');
#wc2.to_file('negative_cloud.png');

wc3 = wordcloud.WordCloud(width=1000, height=1000, random_state=24, 
              max_font_size=200, background_color='white',
                        collocations=False).generate(' '.join(neutralwords))
plt.axis('off');
#wc3.to_file('neutral_cloud.png');

#Plotting some plots
g = sns.FacetGrid(corpus_latin, col='analysis', col_order=['positive', 'negative', 'neutral'])
g.map(sns.countplot, 'length', color='#B24526')
g.set(xticks=np.arange(1,32,2))
g.set_xticklabels(np.arange(1,32,2), size=9);
#plt.savefig('facetgrid_length', bbox_inches='tight', format='eps')

plt.figure(figsize=(16,8))
sns.countplot(data=corpus, x='charlength', color='#B24526')
plt.xticks(np.arange(0,101,20))
plt.xlabel('charlength', labelpad=10)
plt.ylabel('count', labelpad=10)
plt.title('Message length in characters', pad=10);
#plt.savefig('message_length_char', bbox_inches='tight', format='eps')
