# /* Copyright (C) Kannan Sekar Annu Radha - All Rights Reserved
#  * Unauthorized copying of this file, via any medium is strictly prohibited
#  * Proprietary and confidential
#  * Written by Kannan Sekar Annu Radha <kannansekara@gmail.com>, November 2019
#  */ NHS DIGITAL MRS PRIYA BASKER AND MR JOHNATHAN HOPE 
# Innovative uses of Data team NHS DIGITAL

import numpy as np
import pandas as pd
import multiprocessing
import os
import re

import nltk
from nltk.corpus import stopwords

from gensim.models import Word2Vec

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy import spatial

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# For saving embeddings and metadata for TensorFlow Embedding Projector
import tensorflow as tf

# Load the data
print("Loading data from 'd_icd_diagnoses.csv'...")
icdf2 = pd.read_csv('d_icd_diagnoses.csv')
print("Data loaded. First 5 rows:")
print(icdf2.head())

# Download NLTK stopwords
print("Downloading NLTK stopwords...")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print("Stopwords downloaded.")

# Function to clean and tokenize text
def clean_text(text):
    # Remove non-alphabetic characters and lowercase the text
    text = re.sub("[^a-zA-Z]", " ", text)
    words = text.lower().split()
    # Remove stopwords
    words = [w for w in words if w not in stop_words]
    return words

# Clean the 'long_title' column
print("Cleaning text in 'long_title' column...")
icdf2['cleaned_long_title'] = icdf2['long_title'].apply(clean_text)
print("Text cleaned. First 5 rows of cleaned text:")
print(icdf2[['long_title', 'cleaned_long_title']].head())

# Create the raw corpus
raw_corpus = icdf2['cleaned_long_title'].tolist()

# Set hyperparameters for Word2Vec
num_features = 100      # Embedding vector size
min_word_count = 1      # Minimum word count
num_workers = multiprocessing.cpu_count()  # Number of threads to run in parallel
context_size = 7        # Context window size
downsampling = 1e-3     # Downsample setting for frequent words
seed = 1                # Seed for the RNG

# Build the Word2Vec model
print("Building Word2Vec model with vector size:", num_features)
icd2vec = Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    vector_size=num_features,  # Use vector_size instead of size
    min_count=min_word_count,
    window=context_size,
    sample=downsampling,
)

# Build vocabulary and train the model
print("Building vocabulary...")
icd2vec.build_vocab(raw_corpus)
print("Vocabulary size:", len(icd2vec.wv))

print("Training Word2Vec model...")
icd2vec.train(raw_corpus, total_examples=icd2vec.corpus_count, epochs=10)
print("Word2Vec model trained.")

# Save the model
if not os.path.exists("trained"):
    os.makedirs("trained")
icd2vec.save(os.path.join("trained", "icd2vec.model"))
print("Word2Vec model saved.")

# Load the model
icd2vec = Word2Vec.load(os.path.join("trained", "icd2vec.model"))
print("Word2Vec model loaded.")

# Save vectors and metadata for the TensorFlow Embedding Projector
def save_embeddings_for_projector(model, output_path):
    # Save the word vectors (embeddings)
    vec_file = os.path.join(output_path, 'vectors.tsv')
    meta_file = os.path.join(output_path, 'metadata.tsv')
    
    with open(vec_file, 'w', encoding='utf-8') as vec_out, open(meta_file, 'w', encoding='utf-8') as meta_out:
        for word in model.wv.index_to_key:
            vec_out.write('\t'.join([str(x) for x in model.wv[word]]) + "\n")
            meta_out.write(word + "\n")
    
    print(f"Word vectors saved to {vec_file}")
    print(f"Metadata saved to {meta_file}")

# Create the output directory for embeddings
output_path = 'embedding_projector'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Save the embeddings
save_embeddings_for_projector(icd2vec, output_path)

print("Embedding projector data saved.")

# Function to get sentence vector by averaging word vectors
def get_sentence_vector(sentence, model):
    # Filter out words not in the vocabulary
    words_in_vocab = [word for word in sentence if word in model.wv.key_to_index]
    if words_in_vocab:
        return np.mean(model.wv[words_in_vocab], axis=0)
    else:
        return np.zeros(model.vector_size)

# Apply the function to create vectors for each ICD code description
print("Generating sentence vectors for each cleaned long title...")
icdf2['vector'] = icdf2['cleaned_long_title'].apply(lambda x: get_sentence_vector(x, icd2vec))
print("First 5 vectors generated:")
print(icdf2['vector'].head())

# Set 'icd9_code' as the index for easy mapping
icdf2.set_index('icd9_code', inplace=True)

# Load diagnosis data
print("Loading diagnosis data from 'diagnoses_icd.csv'...")
diagdf = pd.read_csv("diagnoses_icd.csv")
print("Diagnosis data loaded. First 5 rows:")
print(diagdf.head())

# Map vectors to diagdf based on 'icd9_code'
print("Mapping vectors to diagnosis dataframe based on 'icd9_code'...")
diagdf['vector'] = diagdf['icd9_code'].map(icdf2['vector'])

# Drop rows without vectors
diagdf.dropna(subset=['vector'], inplace=True)
print("Rows with missing vectors dropped. Remaining data size:", diagdf.shape)

# Map ICD codes to integer labels
print("Mapping ICD-9 codes to integer labels...")
label_encoder = LabelEncoder()
diagdf['icd9_code_label'] = label_encoder.fit_transform(diagdf['icd9_code'])

# Prepare feature matrix X and target vector y
print("Preparing feature matrix X and target vector y...")
X = np.vstack(diagdf['vector'].values)
y = diagdf['icd9_code_label'].values

# One-hot encode the labels
num_classes = len(label_encoder.classes_)
y = to_categorical(y, num_classes=num_classes)

# Reshape X for LSTM input (samples, timesteps, features)
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Split into training and test sets
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data split complete. Training size:", X_train.shape, ", Test size:", X_test.shape)

# Define the LSTM model for classification
print("Building the LSTM model...")
model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))  # Output layer for classification
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
print("Training the LSTM model...")
history = model.fit(
    X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test)
)
print("Model training complete.")

# Make predictions
print("Making predictions on the test set...")
y_pred_prob = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_prob, axis=1)

# Map predicted labels back to ICD codes
predicted_icd_codes = label_encoder.inverse_transform(y_pred_classes)

# Actual ICD codes
y_test_classes = np.argmax(y_test, axis=1)
actual_icd_codes = label_encoder.inverse_transform(y_test_classes)

# Compare predictions with actual ICD codes
print("Comparing predicted and actual ICD-9 codes...")
results = pd.DataFrame({
    'Actual_ICD9_Code': actual_icd_codes,
    'Predicted_ICD9_Code': predicted_icd_codes
})
print("First 5 predictions vs actual results:")
print(results.head())

# Optionally, print the descriptions for each prediction
print("Printing the actual and predicted ICD-9 codes with their descriptions...")
for idx in range(len(results)):
    actual_code = results.loc[idx, 'Actual_ICD9_Code']
    predicted_code = results.loc[idx, 'Predicted_ICD9_Code']
    actual_desc = ' '.join(icdf2.loc[actual_code]['cleaned_long_title'])
    predicted_desc = ' '.join(icdf2.loc[predicted_code]['cleaned_long_title'])
    print(f"Actual ICD-9 Code: {actual_code}, Description: {actual_desc}")
    print(f"Predicted ICD-9 Code: {predicted_code}, Description: {predicted_desc}\n")
