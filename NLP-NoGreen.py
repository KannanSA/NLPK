from __future__ import absolute_import, division, print_function

import codecs
import collections
import csv
import glob
import itertools
import logging
import multiprocessing
import os
import pprint
import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.manifold
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import gzip
from nltk.tokenize import word_tokenize 
from collections import Counter
from string import punctuation
import gensim
from gensim.models import Doc2Vec
from gensim.models.word2vec import LineSentence



diagnoses_filenames = sorted(glob.glob("icd*.csv"))

print(diagnoses_filenames)
print("Found ICD Descriptors")

nltk.download("punkt")
nltk.download("stopwords")

#process data
#init raw unicode
corpus_raw = u""

#read diag, open in utf8, add to corpus_raw

for diagnoses_filename in diagnoses_filenames:
    print("Reading" '{0}'.format(diagnoses_filenames))
    with codecs.open(diagnoses_filename, "r", "utf-8") as diagnoses_file:
        corpus_raw += diagnoses_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()

print(corpus_raw)


## murdo df tutorial
icdf = pd.read_csv("icd1.csv")
icdf["short_title"] = icdf["short_title"].apply(lambda x: x.split())
raw_icdf = icdf["short_title"].to_list()

##more dataframe
icdf2 = pd.read_csv('d_icd_diagnoses.csv')
icdf2['long_title'] =icdf2['long_title'].apply(lambda x: x.split())
raw_icdf2 = icdf2['long_title'].to_list()

#icdlong in tuple
Row_list = []
for rows in icdf2.itertuples():
    my_list =[rows.icd9_code, rows.short_title, rows.long_title]
    Row_list.append(my_list)
print(my_list)

icdf2["sentence"]= icdf2["long_title"].apply(lambda x: ','.join(x).replace(',', ' '))
print(icdf2["sentence"])


raw_corpus = icdf2["long_title"].to_list()
print(raw_corpus)

stopWords = set(stopwords.words('english'))

#preprocessing


example = "my name jeff"
tokenized = word_tokenize(example)
print(tokenized)

corp = list(itertools.chain.from_iterable(raw_icdf))
print(corp)

#Pre processing
##test

uniqueWords = [] 
for i in corp:
      if not i in uniqueWords:
          uniqueWords.append(i)

print(uniqueWords)

#uniqueWords from all words in raw_icdf with stop words removed
stopWords = set(stopwords.words('english'))

cleanedVocab = []
for w in uniqueWords:
    if w not in stopWords:
        cleanedVocab.append(w)

## text 

#ONCE we have vectors
#step 3 - build model
#3 main tasks that vectors help with
#DISTANCE, SIMILARITY, RANKING

# Dimensionality of the resulting word vectors.
#more dimensions, more computationally expensive to train
#but also more accurate
#more dimensions = more generalized
num_features = 300
# Minimum word count threshold.
min_word_count = 3

# Number of threads to run in parallel.
#more workers, faster we train
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 7

# Downsample setting for frequent words.
#0 - 1e-5 is good for this
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
#random number generator
#deterministic, good for debugging
seed = 1 

model = Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling

)

model.build_vocab(raw_icdf)

print("Word2Vec vocabulary length:", len(model.wv.vocab))


w1 = "TB"
model.wv.most_similar (positive=w1)


print(raw_icdf)
print(icdf["short_title"])




####### test
example = "my name jeff"

testm = gensim.models.Word2Vec(icdf2["sentence"], size=150, window=10, min_count=2, workers=10)
testm.train(example, total_examples=100, total_words = 100, epochs = 10)




































