from __future__ import absolute_import, division, print_function

import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re

import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

%pylab inline

#stopwords at unnecessary words
#tokenization into sentences, punkt: pretrained tokenizer
nltk.download("punkt")
#stopwords and and the
nltk.download("stopwords")

#get icd diag and corresponding descriptions short_title

diagnoses_filenames = sorted(glob.glob("/*.csv"))

print("Found ICD Descriptors")

#process data
#init raw unicode
corpus_raw = u""

#read diag, open in utf8, add to corpus_raw
for diagnoses_filename in diagnoses_filenames:
    print("Reading" '{0}'...".format(diagnoses_filenames))
    with codecs.open(diagnoses_filename, "r", "utf-8") as diagnoses_file:
        corpus_raw += diagnoses_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()

#tokenization   save trained model here
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#tokenize into sentences
raw_sentences = tokenizer.tokenize(corpus_raw)

#convert into word list
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]", " ", raw)
    words = clean.split()
    return words

#for each sentance, sentences where each word is tokenized
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) >0:
        sentences.append(sentence_to_wordlist(raw_sentence))

#example
##print(raw_sentence[num])
##print(sentence_to_wordlist(raw_sentences[num]))

#count tokens, each token is a sentence
token_count = sum([len(sentence) for sentence in sentences])
print("The diagnoses corpus contains {0:,} tokens".format(token_count))

#build word2vec model
#hyperparameters

#dimensionality of word vectors
D = 400

min_word_count = 3

#cpu threads available
num_workers =  multiprocessing.cpu_count()

#context window
context_size = 7

#downsample frequent words
#occurance rate from 0 to 1e-3

downsampling = 1e-3

#RNG Seed
seed = 1

icd2vec = w2v.Word2Vec(
    sg = 1,
    seed = seed,
    workers = num_workers,
    size = D,
    min_count = min_word_count,
    window = context_size,
    sample = downsampling

)

icd2vec.build_vocab(sentences)
print("Word2Vec vocabulary length:", len(icd2vec.vocab))

#train model on sentences
icd2vec.train(sentenes)

#save model
if not os.path.exists("trained"):
    os.makedirs("trained")

icd2vec.save(os.path.join("trained"), "icd2vec.w2v"))

#load model
icd2vec = w2v.Word2Vec.load(os.path.joined("trained", "icd2vec.w2v"))

#dimensionality reduction from n to 2 with TNSE
#t-distributed Stochastic Neighbour Embedding ML algorithm, similarity metric
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

#add to matrix
all_word_vectors_matrix = icd2vec.syn0
#train tsne
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

#plot in 2d space to visualise vectors

points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[icd2vec.vocab[word].index])
            for word in icd2vec.vocab
        ]

        ]

    ],
    #come back here to change to ICD csv format
    columns = ["word", "x", "y"]
)

points.head(10)

#plot
sns.set_context("poster")

points.plot.scatter("x","y", s=10, figsize=(20,12))

def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.xx <= x_bounds[1]) &
        (y_bounds[0] <= y_bounds[1])

    ]
    ax = slice.plot.scatter("x", "y", s=35, figsize = (10,8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize = 11)

plot_region(x_bounds=(4.0,4.2), y_bounds=(-0.5,-0.1))
plot(x_bounds=(0,1), y_bounds=(4,4.5))

icd2vec.most_similar("Tubercolosis")

#distance, similarity, ranking
def nearest_similarity_cosmul(start1, end1, end2):
    similarities =  icd2vec.most_similar_cosmul(
        positive = [end2, start1],
        negative = [end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2

#test
#nearest_similarity_cosmul("tb", "cardiac","pulmonary")
