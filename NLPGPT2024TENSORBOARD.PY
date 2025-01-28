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
from tensorflow.keras.utils import plot_model


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

plot_model(
    model, 
    to_file='lstm_arch.png', 
    show_shapes=True, 
    show_layer_names=True
)
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


from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# (A) Generate confusion matrix using numeric label indices
#     If you want to see the matrix in terms of label indices:
cm_labels = confusion_matrix(y_test_classes, y_pred_classes)
print("Confusion matrix (numeric labels):\n", cm_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_labels, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Label Indices)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.pdf", bbox_inches='tight')
plt.show()

# (B) Optionally, generate confusion matrix using actual ICD-9 codes as labels.
#     Be cautious if you have many distinct ICD codes, as the matrix can become large.

unique_codes = sorted(list(set(actual_icd_codes) | set(predicted_icd_codes)))
cm_codes = confusion_matrix(actual_icd_codes, predicted_icd_codes, labels=unique_codes)

print("Confusion Matrix (ICD-9 Codes):")
print(cm_codes)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm_codes,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=unique_codes,
    yticklabels=unique_codes
)
plt.title("Confusion Matrix (ICD-9 Codes)")
plt.xlabel("Predicted ICD-9")
plt.ylabel("Actual ICD-9")
plt.savefig("confusion_matrix_codes.pdf", bbox_inches='tight')
plt.show()

# (C) Print a classification report (numeric labels) for precision, recall, F1
report = classification_report(y_test_classes, y_pred_classes, zero_division=0)
print("Classification Report (numeric labels):\n", report)


# Optionally, print the descriptions for each prediction
print("Printing the actual and predicted ICD-9 codes with their descriptions...")
for idx in range(len(results)):
    actual_code = results.loc[idx, 'Actual_ICD9_Code']
    predicted_code = results.loc[idx, 'Predicted_ICD9_Code']
    actual_desc = ' '.join(icdf2.loc[actual_code]['cleaned_long_title'])
    predicted_desc = ' '.join(icdf2.loc[predicted_code]['cleaned_long_title'])
    print(f"Actual ICD-9 Code: {actual_code}, Description: {actual_desc}")
    print(f"Predicted ICD-9 Code: {predicted_code}, Description: {predicted_desc}\n")

# Optional: Visualize word embeddings with t-SNE
print("Preparing to visualize word embeddings using t-SNE...")
all_word_vectors_matrix = icd2vec.wv.vectors  # For gensim 4.x

# Reduce dimensionality
tsne = TSNE(n_components=2, random_state=0)
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

# Create a DataFrame for visualization
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in zip(icd2vec.wv.index_to_key, all_word_vectors_matrix_2d)
    ],
    columns=["word", "x", "y"]
)

# Plot the embeddings
sns.set_context("poster")
print("Plotting word embeddings...")
points.plot.scatter("x", "y", s=10, figsize=(20, 12))
plt.show()

def create_bilstm_model(vocab_size=5000, embedding_dim=100, max_sequence_length=200):
    """Creates a simple Bidirectional LSTM model."""
    import tensorflow as tf

    model = tf.keras.Sequential()
    
    # Embedding layer
    model.add(tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_sequence_length,
        name="Embedding"
    ))
    
    # Bidirectional LSTM
    model.add(tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=False),
        name="Bidirectional_LSTM"
    ))
    
    # Dense output layer
    model.add(tf.keras.layers.Dense(10, activation='softmax', name="Output_Dense"))
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Make sure to return the model here
    return model

if __name__ == "__main__":
    model = create_bilstm_model()
    # Print a summary
    print(model.summary())

    # Plot the model
    plot_model(
        model,
        to_file='lstm_arch.png',    # Saves diagram as a PNG
        show_shapes=True,           # Show shape of input/output tensors
        show_layer_names=True,       # Show layer names
        expand_nested=False,         # If you have nested models, set to True
        dpi=96                       # Dots per inch
    )
    print("Model architecture saved as lstm_arch.png")
