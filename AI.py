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
from sklearn import preprocessing
from nltk.tokenize import RegexpTokenizer



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
min_word_count = 3

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

#modify epochs training evolution

icd2vec.train(raw_corpus, total_words = len(icd2vec.wv.vocab), total_examples = len(raw_corpus), epochs = 10)
#save trained model
if not os.path.exists("trained"):
    os.makedirs("trained")
icd2vec.save(os.path.join("trained", "icd2vec.w2v"))


###icd2vec.wv.save_word2vec_format('model.bin', binary=True) <- doesn't work
######   "{0}.w2v".format(0 = randint())


#explore trained model
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


############ know it works up to here








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
















######### functions from email
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

##############




##########-------------------##############
########        Now using patient data from patient_diag.csv or diagnoses_icd
### for long_title average words in sentence with TF-IDF
### term frequency–inverse document frequency

def basic_preprocessing(df,notes,label):
   

    df_temp = df.copy(deep = True)

    df_temp = df_temp.rename(index = str, columns = {notes: 'text'})

    df_temp.loc[:, 'text'] = [(x) for x in df_temp['text'].values]

    le = preprocessing.LabelEncoder()

    le.fit(df_temp[label])

    df_temp.loc[:, 'class_label'] = le.transform(df_temp[label])   

    tokenizer = RegexpTokenizer(r'\w+')

    df_temp["tokens"] = df_temp["text"].apply(tokenizer.tokenize)

    return df_temp



def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):

    if len(tokens_list)<1:

        return np.zeros(k)

    if generate_missing:

        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]

    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
        length = len(vectorized)
        summed = np.sum(vectorized, axis=0)
        averaged = np.divide(summed, length)
    return averaged, vector


 

def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False):

    embeddings = clean_questions['tokens'].apply(lambda x: get_average_word2vec(x, vectors,

                                                                                generate_missing=generate_missing))

    return list(embeddings)

           

def w2v(data,notes,label,word2vec):

    df_temp = data.copy(deep = True) 
    df_temp = basic_preprocessing(df_temp,notes,label)

    embeddings = get_word2vec_embeddings(word2vec, df_temp)

    list_labels = df_temp["class_label"].tolist()
   
    return embeddings, list_labels


w2v(icdf2,'sentence','icd9_code', icd2vec)

X,y = w2v(icdf2,'sentence','icd9_code', icd2vec)
print(X,y)

# wordvectors=icd2vec.wv #KeyedVectors Instance gets stored 
# print(wordvectors.syn0)
# icdf2['Csentence'].apply(item for item in lambda x: icd2vec.wv.vocab[item])
# wordvectors.word_vec("tuberculosis")

# my_dict = dict({})
# for idx, key in enumerate(icd2vec.wv.vocab):
#     my_dict[key] = icd2vec.wv[key]
#     print("{0}, {1}".format(0= idx, 1= icd2vec.wv[key])
#     Or my_dict[key] = model.wv.get_vector(key)
#     Or my_dict[key] = model.wv.word_vec(key, use_norm=False)
#     Or my_dict[key] = icd2vec.wv.get_vector(key)
#     Or my_dict[key] = icd2vec.wv.word_vec(key, use_norm=False)



def basic_preprocessing(df,notes,label):
   

    df_temp = df.copy(deep = True)

    df_temp = df_temp.rename(index = str, columns = {notes: 'text'})

    df_temp.loc[:, 'text'] = [(x) for x in df_temp['text'].values]

    le = LabelEncoder()

    le.fit(df_temp[label])

    df_temp.loc[:, 'class_label'] = le.transform(df_temp[label])   

    tokenizer = RegexpTokenizer(r'\w+')

    df_temp["tokens"] = df_temp["text"].apply(tokenizer.tokenize)

    return df_temp

 




 

X, y = icd2vec.w2v(data_select,'APPT_TYPE','NAT_CAT',Word2Vec)

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y, test_size=0.2, random_state=40)

folds = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 40)

clf_w2v = LogisticRegressionCV(cv = folds, solver = 'saga', multi_class = 'multinomial', n_jobs = -1)

clf_w2v.fit(X_train_log, y_train_log)

y_pred = clf_w2v.predict(X_test_log)

 

 

# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.naive_bayes import MultinomialNBX_train, X_test, y_train, y_test = train_test_split(df['Consumer_complaint_narrative'], df['Product'], random_state = 0)
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(X_train)
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)clf = MultinomialNB().fit(X_train_tfidf, y_train)