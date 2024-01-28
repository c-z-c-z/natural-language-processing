#Performing latent Dirichlet allocation (LDA) with gensim on the expanded demographic corpus

import pandas as pd
import spacy

#Keeping some quantitative information, dropping some superfluous columns

to_keep = ['text','species','laff','gender','missing','word_count', 'char_count', 'compound_score', 'subjectivity']
corpus = pd.read_csv('extended_corpus_processed.csv', header=0, usecols=to_keep)

#A quick peek
corpus.head()

nlp = spacy.load('en_core_web_sm')

#Tokenizing each chat message
def tokenize_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop]
    return tokens

#Need to make sure all chat messages are strings as there are some integers in there
corpus['text'] = corpus['text'].apply(str)
corpus['tokens'] = corpus['text'].apply(tokenize_text)

#Using gensim for topic modeling
import gensim
from gensim import corpora

#Map words to integer IDs, then convert this document structure to a bag-of-words format
texts = corpus['tokens'].to_list()
dictionary = corpora.Dictionary(texts)
dict_corpus = [dictionary.doc2bow(text) for text in texts]

from gensim.models import LdaModel, CoherenceModel

#Calculating coherence scores to decide amount of passes to perform through the corpus with the LDA model
for passes in range(1, 11):
    lda_model = LdaModel(dict_corpus, num_topics=10, random_state=114, id2word=dictionary, passes=passes)
    coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print(f'Passes: {passes}, Coherence score: {coherence_score}')

#A coherence score of >0.6 is generally considered good (topics are at least somewhat well-defined and semantically coherent)
#We'll go with 10 passes

lda_model = LdaModel(dict_corpus, num_topics=10, random_state=114, id2word=dictionary, passes=10)

#Checking out the top words for each topic
import pprint
pp = pprint.PrettyPrinter(indent=2)
topics = lda_model.show_topics(num_topics=10, num_words=20, formatted=False)

for t, words in topics:
  print(f"Topic {t + 1}:")
  pp.pprint([(word, round(prob,3)) for word, prob in words])
  print("\n")

#Assigning topic number to individual documents/messages
topic_assignments = []
for d in dict_corpus:
  topic_values = dict(lda_model[d])
  topic_number = max(topic_values.items(), key=lambda x: x[1])[0]
  topic_assignments.append(topic_number+1)

#LDA model holds list for each document with values indicating how well the document corresponds to each topic
#Grabbing maximum value indicating the topic in which the document fits most optimally
#Then looking up corresponding topic number

#Creating a new column in the corpus based on topic assignments
corpus['topic'] = topic_assignments

corpus.head()

#VISUALIZATION

#Quantitative metrics and topics

#Normalizing counts when visualizing because there are huge differences between demographic group sizes

def normalized_counts_pivot(data, category, metric):
  #takes in dataset, demographic category, category whose counts tp normalize
  proportions = data.groupby([category, metric]).size().reset_index(name='count')
  counts = proportions.groupby(category)['count'].sum()
  proportions['proportion'] = round(proportions['count'] / proportions[category].map(counts),3)
  pivot = proportions.pivot(index=category, columns=metric, values='proportion').fillna(0)
  return pivot

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Species
species_pivot = normalized_counts_pivot(corpus, 'species', 'topic')

plt.figure(figsize=(12, 8))
sns.heatmap(species_pivot, annot=True, cmap='crest', fmt='.1%', cbar=False)
plt.title('Topic distribution across species')
plt.show()

#Monkeys stand out for topic 4
#The top word for topic 4 is beans and beans is also the top word for monkeys
#This was uncovered in January 2024 demographic NLP research paper on extended corpus

#Gender
gender_pivot = normalized_counts_pivot(corpus, 'gender', 'topic')
gender_labels = ['m', 'f']

plt.figure(figsize=(12, 8))
sns.heatmap(gender_pivot, annot=True, cmap='crest', fmt='.1%', yticklabels = gender_labels, cbar=False)
plt.title('Topic distribution across gender')
plt.show()
#cbar_kws={'label': 'percentage'}

#Missing Gag track
m_pivot = normalized_counts_pivot(corpus, 'missing', 'topic')
missing_labels = ['nm', 'toon-up', 'trap', 'lure', 'sound', 'drop']

plt.figure(figsize=(12, 8))
sns.heatmap(m_pivot, annot=True, cmap='crest', fmt='.1%', yticklabels = missing_labels, cbar=False)
plt.title('Topic distribution across missing track')
plt.show()

#Splitting dataset into categories for Laff
#Laff is a Toon's health and ranges from 15 to 140
bins = [15,19,29,39,49,59,69,79,89,99,109,119,129,139,140]
laff_labels = ['15-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99','100-109','110-119','120-129','130-139','140']
corpus['laff_range'] = pd.cut(corpus['laff'], bins=bins, labels=laff_labels, include_lowest=True).astype('string')

l_pivot = normalized_counts_pivot(corpus, 'laff_range', 'topic')

plt.figure(figsize=(12, 8))
sns.heatmap(l_pivot, annot=True, cmap='crest', fmt='.1%', yticklabels=laff_labels, cbar=False)
plt.title('Topic distribution across Laff ranges')
plt.show()

#Interestingly, topic 4 is much more popular among Toons with higher Laff
#"beans" is the top word for topic 4
#The frequency of "beans" as a token decreases as Laff increases and is much more popular for lower Laff individuals

#Quantitative metrics

#Message length in words
wl_info = corpus.groupby('topic')['word_count'].mean().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(x='topic', y='word_count', color='navy', alpha=0.7, data=wl_info)
plt.xlabel('Topic', labelpad=10)
plt.ylabel('Message length (words)', labelpad=10)
plt.show()

#Message length in characters
cl_info = corpus.groupby('topic')['char_count'].mean().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(x='topic', y='char_count', color='navy', alpha=0.7, data=cl_info)
plt.xlabel('Topic', labelpad=10)
plt.ylabel('Message length (characters)', labelpad=10)
plt.yticks(np.arange(0,26,5))
plt.show()

#Extreme similarity between the two plots

#Sentiment score
sent_info = corpus.groupby('topic')['compound_score'].mean().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(x='topic', y='compound_score', color='navy', alpha=0.7, data=sent_info)
plt.xlabel('Topic', labelpad=10)
plt.ylabel('Sentiment', labelpad=10)
plt.yticks(np.arange(0,0.26,0.05))
plt.show()

#Subjectivity
subj_info = corpus.groupby('topic')['subjectivity'].mean().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(x='topic', y='subjectivity', color='navy', alpha=0.7, data=subj_info)
plt.xlabel('Topic', labelpad=10)
plt.ylabel('Subjectivity', labelpad=10)
plt.yticks(np.arange(0,0.36,0.05))
plt.show()

#Interactive dashboard
import pyLDAvis
import pyLDAvis.gensim
import pickle

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, dict_corpus, dictionary)
#vis

#Call vis to see interactive dashboard
#Distribution of topics, relationships between topics, most relevant terms for each topic
