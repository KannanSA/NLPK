# /* Copyright (C) Kannan Sekar Annu Radha - All Rights Reserved
#  * Unauthorized copying of this file, via any medium is strictly prohibited
#  * Proprietary and confidential
#  * Written by Kannan Sekar Annu Radha <kannansekara@gmail.com>, November 2019
#  */ NHS DIGITAL MRS PRIYA BASKER AND MR JOHNATHAN HOPE 
# Innovative uses of Data team NHS DIGITAL

from __future__ import absolute_import, division, print_function, unicode_literals

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
from numpy.lib.twodim_base import diag
import regex as re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.manifold
from gensim.models import Word2Vec
import gensim.models.word2vec as w2v

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import gzip
from nltk.tokenize import word_tokenize 
from collections import Counter
from string import punctuation
import gensim
from gensim.models import Doc2Vec
from gensim.models.word2vec import LineSentence
from sklearn.manifold import TSNE
from gensim.models import Word2Vec, KeyedVectors   

import numpy as np

import tensorflow
import keras
import tensorflow as tf

from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Sequential
# from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D, MaxPooling1D


nltk.download('stopwords')


## murdo df tutorial
icdf = pd.read_csv("icd1.csv")
icdf["short_title"] = icdf["short_title"].apply(lambda x: x.split())
raw_icdf = icdf["short_title"].to_list()

##more dataframe
icdf2 = pd.read_csv('d_icd_diagnoses.csv')
icdf2['long_title'] =icdf2['long_title'].apply(lambda x: x.split())
raw_icdf2 = icdf2['long_title'].to_list()

icdf2.head(10)

################# new vocab for long list
icdf2["sentence"]= icdf2["long_title"].apply(lambda x: ','.join(x).replace(',', ' '))
icdf2["sentence"]
# print(icdf2["sentence"])

raw_corpus = icdf2["long_title"].to_list()
print(raw_corpus)


#cleaned dataframe
icdf2['Csentence'] = icdf2['long_title'].apply(lambda x: ','.join(x).replace(',', ' '))
icdf2['Csentence'] = icdf2['Csentence'].apply(lambda x: re.sub("[^a-zA-Z]"," ", x))
icdf2['Csentence'] = icdf2['Csentence'].apply(lambda x: x.lower())
stop = stopwords.words('english')
icdf2['Csentence'].apply(lambda x: x.split())
#df['response'] = df['response'].apply(lambda x: [item for item in x.split() if item not in stop])
icdf2['Csentence'] = icdf2['Csentence'].apply(lambda x: [item for item in x.split() if item not in stop])

####change corpus to anything  unclean corpus was raw_corpus########
##########################
##########################
##########################
raw_corpus = icdf2['Csentence']
##########################
##########################
##########################


icdf2['Csentence']
#print(len(icdf2['Csentence']))
print("Corpus is now {0} characters long".format(len(icdf2['Csentence'])))

icdf2.head(10)


#### corpus is sentence in ipy train it on
# for corpus in raw_corpus:
#     print(corpus)

########### icd2vec
#define hyperparameters

# Dimensionality
num_features = 3

#
# Minimum word count threshold.
min_word_count = 1

# Number of threads to run in parallel.
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 7

# Downsample setting for frequent words.
#rate 0 and 1e-5 
#how often to use
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
seed = 1

icd2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling,
)

#----------- important don't delete-----------#
# if not os.path.exists("TENSORS"):
#     os.makedirs("TENSORS")
# icd2vec.save(os.path.join("TENSORS", "icd2vec.w2v"))
model = Word2Vec(raw_corpus)
model.wv.save_word2vec_format('cleanmodel_name')
# ##python -m gensim.scripts.word2vec2tensor --input model_name --output model_name
###^^^^ run command in TENSORS file in your terminal

icd2vec.build_vocab(raw_corpus)
print("Word2Vec vocabulary length:", len(icd2vec.wv.vocab))

# unique word set
index2word_set = set(icd2vec.wv.index2word)
#print(index2word_set)
len(index2word_set)

#modify epochs training evolution total_words = len(icd2vec.wv.vocab),

icd2vec.train(raw_corpus, total_words =78448 , total_examples = len(raw_corpus), epochs = 10)
#save trained model
if not os.path.exists("trained"):
    os.makedirs("trained")
icd2vec.save(os.path.join("trained", "icd2vec.w2v"))


###icd2vec.wv.save_word2vec_format('model.bin', binary=True) <- doesn't work
######   "{0}.w2v".format(0 = randint())


#explore trained model   #load model
icd2vec = w2v.Word2Vec.load(os.path.join("trained", "icd2vec.w2v"))

#compress word vectors into 2d
#dimensionality reduction from n to 2
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
all_word_vectors_matrix = icd2vec.wv.syn0


#####DeprecationWarning  eprecated `syn0` Attribute will be removed in 4.0.0, use self.vectors instead).




#train t-sne
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
#plot picture

points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[icd2vec.wv.vocab[word].index])
            for word in icd2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)

points.head(10)
sns.set_context("poster")
points.plot.scatter("x", "y", s=10, figsize=(20, 12))

def plot_region(x_bounds, y_bounds, points):
    '''
    plot a sub-region of words in a tSNE reduction, for a dataframe: points
    '''
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) &
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1]) 
    ]
     
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)








# ############ know it works up to here








def display_closestwords_tsnescatterplot(model, word, vec_size):
    '''
    tSNE visualising Word2Vec function.
    '''
    arr = np.empty((0, vec_size), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.title('Most similar word vectors in the generated embeddings for term: "' \
        + word + '"')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.xlabel("tSNE dimension 1")
    plt.ylabel("tSNE dimension 2")
    plt.show()

print (points.head(10))
print (points.tail(10))
plot_region(x_bounds=(0, 10), y_bounds=(0, 10), points=points)
















# ######## functions from email
# The function below used TSNE - Distributed Stochastic neighbouthood embedding method.
# This is a dimensionality reduction methos which reduces the high dimensional data into 2d or 3d space


def display_closestwords_tsnescatterplot(icd2vec, word):
    
    arr = np.empty((0,300), dtype='f')
    word_labels = [word]

    # get close words
    #   close_words = model.similar_by_word(word)
    #### top n to change output
    close_words=icd2vec.most_similar(positive=word,topn=10)

    
    # add the vector for each of the closest words to the array    
    #   print(np.array([model[word[1]]]))
    arr = np.array([icd2vec[word[0]]])
    for wrd_score in close_words:
        wrd_vector = icd2vec[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)
 
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()

#drop_duplicates : Split string by ', ', drop duplicates and join back.

def drop_duplicates(row):
    words = row.split(', ')
    return ', '.join(np.unique(words).tolist())

#unique : picks the unique values from the list
def unique(list1):  

    # intilize a null list
    unique_list = list()
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x)
    return unique_list
     
#     for x in unique_list: 
#         print(x)




display_closestwords_tsnescatterplot(icd2vec, ['tuberculosis','tubercle','nervous'])


display_closestwords_tsnescatterplot(icd2vec, ['psychosis','depression','nervous'])




##############




##########-------------------##############
########        Now using patient data from patient_diag.csv or diagnoses_icd
### for long_title average words in sentence with TF-IDF
### term frequency–inverse document frequency
icdf2.head(10)

icd2vec.wv["tuberculosis"]


icdf2['vector'] = icdf2['Csentence'].apply(lambda x: sum(icd2vec.wv[x]))

# icdf2['avg'] = icdf2['Csentence'].apply(lambda x: np.mean(icd2vec.wv[x]))

icdf2.head(10)
### generates unique vectors for each ICD token

icdf2['vector']


###another for loop for each sentences
# for x in icdf2['Csentence']:
#     for i in x:
#             icd2vec.wv[x]
#             sum(icd2vec.wv[x])
#             print(vect)


# for i in icdf2.Csentence[0]:
#     print(i)

icdf2.head(10)

###diagnoses dataframe
#######retrieve vector by using icd index
icdf2.set_index('icd9_code', inplace=True)


diagdf = pd.read_csv("diagnoses_icd.csv")
diagdf.head(10)

##df.set_index(KEY).to_dict()[VALUE]
icdf2
dict = icdf2.to_dict()['vector']
print(dict)

icdf2['vector']['01723']
icdf2['vector']
icdf2['vector']['01716']


diagdf['vector'] = diagdf['icd9_code'].apply(lambda x: dict.get(x))
diagdf['vector']




#unique word set
# # index2word_set = set(icd2vec.wv.index2word)
# # # print(index2word_set)
# # len(index2word_set)


#joined cleaned sentences
# # # icdf2['JCsentence'] = icdf2['Csentence'].apply(lambda x: ' '.join(x)) 
# # # icdf2['JCsentence']
###3tokenizes sentences is icdf2['Csentence']

icdf2['vector'][5672].shape



###########
########$$$$$$$$%%%%%%%%^^^^^^^ dimensionality reduction
#compress word vectors into 2d
#dimensionality reduction from n to 2
# # # tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
# # # icd_vectors_matrix = 


#####DeprecationWarning  eprecated `syn0` Attribute will be removed in 4.0.0, use self.vectors instead).




# #train t-sne
# all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
# #plot picture




###################$$$$$$$$$$$$$$$$$$$$$$$$$$$$################$$$$$$$$$$$$$$$$$$$$$$$########
################          LSTM Long Short Term Memory RRN in Keras


icdf2.head(10)
diagdf.head(10)


# diagdf['2d'] = diagdf['vector'].apply(lambda x: )

# Load the Data


##### CNN-RNN  Convolutional neural, LSTM
# (X_train, y_train) = (diagdf['vector'].values, diagdf['icd9_code'].values)
# np.asarray(x).astype('float32')
diagdf.dtypes
print(np.asarray(diagdf['vector'][0]).astype("float32"))
for i in diagdf['vector']:
    x = np.asarray(i).astype('float32')
# x = np.stack(x, axis=0) 
for i in diagdf['seq_num']:
    y = np.asarray(i).astype('float32')

# x = np.asarray(diagdf['vector']).astype('float32')

# (X_train, y_train) = (np.asarray(diagdf['vector'].astype("float32")), np.asarray(diagdf['icd9_code']).astype("float32"))
data = (x.reshape(1,-1), y.reshape(1,-1))
model.fit(data, data, epochs=1)

model.fit( x=data[0] , y=data[1], batch_size=10 , epochs=10 , verbose=1 , validation_data = (data[0],data[1]))

X_train = x.reshape(-1, 1, 3)
X_test  = x.reshape(-1, 1, 3)
y_train = y.reshape(-1, 1, 1)
y_test = y.reshape(-1, 1, 1)

model = Sequential()
model.add(LSTM(100, input_shape=(1, 3), return_sequences=True))
model.add(LSTM(5, input_shape=(1, 3), return_sequences=True))
model.compile(loss="mean_absolute_error", optimizer="adam", metrics= ['accuracy'])

history = model.fit(X_train,y_train,epochs=100, validation_data=(X_test,y_test))
model.save('kannan')
model = load_model('kannan')

yhat = model.predict(X_train, verbose=0)
print(yhat)
# model.fit(X_train, y_train, batch_size = 100, epochs = 20, verbose = 1)
history = model.fit(
    x,
    y,
    batch_size=64,
    epochs=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)
### Custom neural layer Ajit Patra Rhodes scholar suggestion Jenner Institute, University of Oxford
# TensorFlow and tf.keras


