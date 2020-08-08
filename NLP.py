# /* Copyright (C) Kannan Sekar Annu Radha - All Rights Reserved
#  * Unauthorized copying of this file, via any medium is strictly prohibited
#  * Proprietary and confidential
#  * Written by Kannan Sekar Annu Radha <kannansekara@gmail.com>, November 2019
#  */ NHS DIGITAL MRS PRIYA BASKER AND MR JOHNATHAN HOPE 
# Innovative uses of Data team NHS DIGITAL

#ICD2VEC NLPK

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
num_features = 400

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
ICD_tensors =diagdf['vector'].reshape
print(ICD_tensors)

diagdf['vector'][0].reshape(4,100)

data_dim = 400
timesteps = 8
num_classes = 2

model = Sequential()
model.add(LSTM(30, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 30
model.add(LSTM(30, return_sequences=True))  # returns a sequence of vectors of dimension 30
model.add(LSTM(30))  # return a single vector of dimension 30
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()
(X_train, y_train) = (diagdf['vector'], diagdf['icd9_code'])

model.fit(X_train, y_train, batch_size = 400, epochs = 20, verbose = 1)



### Custom neural layer Ajit Patra Rhodes scholar suggestion Jenner Institute, University of Oxford
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# 'load data
fashion_mnist = diagdf['vector'].reshape
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# init
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images / 255.0
test_images = test_images / 255.0









###custom layer lambda
###reshape transpose
from keras.layers import Reshape

## Need to write custom neural layer to transfor 400,1 tensors. Manifold learning
yote = Sequential()
yote.add(TimeDistributed(Conv1D(filters=5, kernel_size=3, activation='relu'), batch_input_shape=(24,None,24,1)))
yote.add(TimeDistributed(MaxPooling1D(pool_size=2)))
yote.add(TimeDistributed(Flatten()))
yote.add(LSTM(50, stateful=True, return_sequences=True))
yote.add(LSTM(10, stateful=True))
yote.add(Dense(24))
yote.compile(optimizer='adam', loss='mse', metrics= ['mae', 'mape', 'acc'])

yote.summary()

dataVar_tensor = tf.constant(diagdf['vector'][0], dtype = tf.float32, shape=[400,1])
for x in dataVar_tensor:
    print(x)

# a = np.array(diagdf[])
diagdf['vector'].values[0]
ICD_tensors =diagdf['vector'].values


tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
tensors_matrix_2d = tsne.fit_transform(ICD_tensors)
print(dataVar_tensor)







num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length



# yote = Sequential()
# yote.add(Reshape((400, 1), input_shape=(400,1)))
# yote.add(TimeDistributed(Conv1D(filters=5, kernel_size=3, activation='relu'), batch_input_shape=(24,None,24,1)))
# yote.add(TimeDistributed(MaxPooling1D(pool_size=2)))
# yote.add(TimeDistributed(Flatten()))
# yote.add(LSTM(50, stateful=True, return_sequences=True))
# yote.add(LSTM(10, stateful=True))
# yote.add(Dense(24))
# yote.compile(optimizer='adam', loss='mse', metrics= ['mae', 'mape', 'acc'])

# yote.summary()



yote = Sequential()
yote.add(Reshape((3, 3), input_shape=(400,1)))
yote.add(TimeDistributed(Conv1D(filters=5, kernel_size=3, activation='relu'), batch_input_shape=(24,None,24,1)))
yote.add(TimeDistributed(MaxPooling1D(pool_size=2)))
yote.add(TimeDistributed(Flatten()))
yote.add(LSTM(50, stateful=True, return_sequences=True))
yote.add(LSTM(10, stateful=True))
yote.add(Dense(24))
yote.compile(optimizer='adam', loss='mse', metrics= ['mae', 'mape', 'acc'])

yote.summary()



















def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)

print(generateData())


batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])












# batch_size = 64
# # Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
# # Each input sequence will be of size (28, 28) (height is treated like time).
# input_dim = 28

# units = 64
# output_size = 10  # labels are from 0 to 9

# # Build the RNN model
# def build_model(allow_cudnn_kernel=True):
#   # CuDNN is only available at the layer level, and not at the cell level.
#   # This means `LSTM(units)` will use the CuDNN kernel,
#   # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
#   if allow_cudnn_kernel:
#     # The LSTM layer with default options uses CuDNN.
#     lstm_layer = tf.keras.layers.LSTM(units, input_shape=(None, input_dim))
#   else:
#     # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
#     lstm_layer = tf.keras.layers.RNN(
#         tf.keras.layers.LSTMCell(units),
#         input_shape=(None, input_dim))
#   model = tf.keras.models.Sequential([
#       lstm_layer,
#       tf.keras.layers.BatchNormalization(),
#       tf.keras.layers.Dense(output_size, activation='softmax')]
#   )
#   return model


# mnist = tf.keras.datasets.mnist

# print(minst)
# mnist.load_data()

# # (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # # #compress word vectors into 2d
# # # #dimensionality reduction from n to 2
# # # tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
# # # all_word_vectors_matrix = icd2vec.wv.syn0


# # # #####DeprecationWarning  eprecated `syn0` Attribute will be removed in 4.0.0, use self.vectors instead).




# # # #train t-sne
# # # all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
# # # #plot picture

# # # points = pd.DataFrame(
# # #     [
# # #         (word, coords[0], coords[1])
# # #         for word, coords in [
# # #             (word, all_word_vectors_matrix_2d[icd2vec.wv.vocab[word].index])
# # #             for word in icd2vec.wv.vocab
# # #         ]
# # #     ],
# # #     columns=["word", "x", "y"]
# # dimensionality reduction from n to 2  #####how do you do dimensionality reduction from a list of vectors in panadas df?

# icdf2.head(10)
# diagdf.head(10)
# diagdf['vector']


# print(all_word_vectors_matrix_2d)

# # tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

# # diagdf['2d'] =  diagdf['vector'].apply(lambda x: tsne.fit_transform(x))
# # diagdf['2d']

# tf.enable_eager_execution()

# training_df: pd.DataFrame = pd.DataFrame(
#     data={
#         'feature1': np.random.rand(10),
#         'feature2': np.random.rand(10),
#         'feature3': np.random.rand(10),
#         'target': np.random.randint(0, 3, 10)
#     }
# )
# features = ['feature1', 'feature2', 'feature3']
# print(training_df)

# training_dataset = (
#     tf.data.Dataset.from_tensor_slices(
#         (
#             tf.cast(training_df[features].values, tf.float32),
#             tf.cast(training_df['target'].values, tf.int32)
#         )
#     )
# )

# for features_tensor, target_tensor in training_dataset:
#     print(f'features:{features_tensor} target:{target_tensor}')





# (x_train, y_train), (x_test, y_test) = diagdf['vector']

# x_train, x_test = x_train / 255.0, x_test / 255.0
# sample, sample_label = x_train[0], y_train[0]

######LSTM

# target = diagdf.pop('vector')
# dataset = tf.data.Dataset.from_tensor_slices((diagdf.values, target.values))




























# batch_size = 32
# # batch_size sequences of length 10 with 2 values for each timestep
# input = get_batch(X, batch_size).reshape([batch_size, 10, 2])
# # Create LSTM cell with state size 256. Could also use GRUCell, ...
# # Note: state_is_tuple=False is deprecated;
# # the option might be completely removed in the future
# cell = tf.nn.rnn_cell.LSTMCell(256, state_is_tuple=True)
# outputs, state = tf.nn.dynamic_rnn(cell,
#                                    input,
#                                    sequence_length=[10]*batch_size,
#                                    dtype=tf.float32)

# predictions = tf.contrib.layers.fully_connected(state.h,
#                                                 num_outputs=1,
#                                                 activation_fn=None)
# loss = get_loss(get_batch(Y).reshape([batch_size, 1]), predictions)
























