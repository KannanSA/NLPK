#zoom



#don't know how to use glob
# diagnoses_filenames = sorted(glob.glob("icd*.csv"))

# print(diagnoses_filenames)
# print("Found ICD Descriptors")

#only needs to run once
# nltk.download("punkt")
# nltk.download("stopwords")

# #process data
# #init raw unicode
# corpus_raw = u""

# #read diag, open in utf8, add to corpus_raw

# for diagnoses_filename in diagnoses_filenames:
#     print("Reading" '{0}'.format(diagnoses_filenames))
#     with codecs.open(diagnoses_filename, "r", "utf-8") as diagnoses_file:
#         corpus_raw += diagnoses_file.read()
#     print("Corpus is now {0} characters long".format(len(corpus_raw)))
#     print()

# print(corpus_raw)







# ######### tsne plot
# def tsne_plot(model):
#     "Creates and TSNE model and plots it"
#     labels = []
#     tokens = []

#     for word in model.wv.vocab:
#         tokens.append(model[word])
#         labels.append(word)
    
#     tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
#     new_values = tsne_model.fit_transform(tokens)

#     x = []
#     y = []
#     for value in new_values:
#         x.append(value[0])
#         y.append(value[1])
        
#     plt.figure(figsize=(16, 16)) 
#     for i in range(len(x)):
#         plt.scatter(x[i],y[i])
#         plt.annotate(labels[i],
#                      xy=(x[i], y[i]),
#                      xytext=(5, 2),
#                      textcoords='offset points',
#                      ha='right',
#                      va='bottom')
#     plt.show()

# tsne_plot(icd2vec)

# icd2vec.most_similar('tuberculosis')














# #zoom in
# def plot_region(x_bounds, y_bounds):
#     slice = points[
#         (x_bounds[0] <= points.x) &
#         (points.x <= x_bounds[1]) & 
#         (y_bounds[0] <= points.y) &
#         (points.y <= y_bounds[1])
#     ]
    
#     ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
#     for i, point in slice.iterrows():
#         ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

# #related datapoints
# plot_region(x_bounds=(4.0, 4.2), y_bounds=(-0.5, -0.1))


# #related datapoints
# plot_region(x_bounds=(0, 1), y_bounds=(4, 4.5))

# #explore semantic similarity
# icd2vec.most_similar("cardiac")

# icd2vec.most_similar("pulmonary")

# icd2vec.most_similar("dystolic")


# #linear relationship between data

# def nearest_similarity_cosmul(start1, end1, end2):
#     similarities = icd2vec.most_similar_cosmul(
#         positive=[end2, start1],
#         negative=[end1]
#     )
#     start2 = similarities[0][0]
#     print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
#     return start2

# nearest_similarity_cosmul("osteoporosis", "rheumatoid", "cardiovascular")



# # stopWords = set(stopwords.words('english'))


# # #preprocessing

# # example = "my name jeff"
# # tokenized = word_tokenize(example)
# # print(tokenized)

# # corp = list(itertools.chain.from_iterable(raw_icdf))
# # print(corp)



# # ## text 

# # #ONCE we have vectors
# # #step 3 - build model
# # #3 main tasks that vectors help with
# # #DISTANCE, SIMILARITY, RANKING

# # # Dimensionality of the resulting word vectors.
# # #more dimensions, more computationally expensive to train
# # #but also more accurate
# # #more dimensions = more generalized
# # num_features = 300
# # # Minimum word count threshold.
# # min_word_count = 3

# # # Number of threads to run in parallel.
# # #more workers, faster we train
# # num_workers = multiprocessing.cpu_count()

# # # Context window length.
# # context_size = 7

# # # Downsample setting for frequent words.
# # #0 - 1e-5 is good for this
# # downsampling = 1e-3

# # # Seed for the RNG, to make the results reproducible.
# # #random number generator
# # #deterministic, good for debugging
# # seed = 1 

# # model = Word2Vec(
# #     sg=1,
# #     seed=seed,
# #     workers=num_workers,
# #     size=num_features,
# #     min_count=min_word_count,
# #     window=context_size,
# #     sample=downsampling

# # )

# # model.build_vocab(raw_icdf)

# # print("Word2Vec vocabulary length:", len(model.wv.vocab))


# # w1 = "TB"
# # model.wv.most_similar (positive=w1)


# # print(raw_icdf)
# # print(icdf["short_title"])



# # ####### test
# # example = "my name jeff"

# # testm = gensim.models.Word2Vec(icdf2["sentence"], size=150, window=10, min_count=2, workers=10)
# # testm.train(example, total_examples=100, total_words = 100, epochs = 10)


# # #LineSentence(icdf2['sentence'])




















# # testm = gensim.models.Doc2Vec(size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025, iter=20)
# # # testm.build_vocab(it)
# # # testm.train(it, epochs=model.iter, total_examples=model.corpus_count)
# # def train_model(corpus, size=200, window=5, workers=3, model_path=None,
# #                 word_freq=None, corpus_count=None):
# #     """Train using Skipgram model.

# #     Args:
# #         corpus (str):       file path of corpus
# #         size (int):         embedding size (default=200)
# #         window (int):       window size (default=5)
# #         workers (int):      number of workers (default=3)
# #         model_path (str):   file path of model we want to update
# #         word_freq (dict):   dictionary of word frequencies
# #         corpus_count (int): corpus size

# #     Returns:
# #         Word2Vec: word2vec model
# #     """































# # #####idk how to use stopwords
# # # clean_corpus = []
# # # for corpus in raw_corpus:
# # #      for word in corpus:
# # #          if word.lower() not in stopWords:
# # #              temp_corpus = []
# # #              temp_corpus.append(word.lower)
# # # clean_corpus = temp_corpus
# # # print(clean_corpus)



# # # uniqueCorpus = [] 
# # # for i in raw_corpus:
# # #     for j in i:
# # #         jeffC = []
# # #         jeff = j.strip(',').lower()
# # #         jeffC.append(jeff)
# # #         print(jeffC)
# # # uniqueCorpus = set(jeffC)
# # # print(uniqueCorpus)

# # # stopW = set(stopwords.words('english'))
# # # clean_corpi = []

# # # # # stoplist = set(stopwords.words('english') + list(punctuation))

# # # # # texts = df['long_title'].str.lower()
# # # # # print(texts)

# # # # # word_counts = Counter(word_tokenize('\n'.join(texts)))

# # # # # word_count.most_common()


# # # # # word_tokens = word_tokenize(i)





# # # # print(icdf2)
# # #print [[' '.join(i)] for rows in icdf2['long_title'] for rows in icdf2]





# # # # sentences_icd = []
# # # # for i in icdf["short_title"]:
# # # #     sentences_icd.append(join(i))


# # # def read_input(input_file):
# # #     """This method reads the input file which is in gzip format"""
    
# # #     logging.info("reading file {0}...this may take a while".format(input_file))
    
# # #     with gzip.open (input_file, 'rb') as f:
# # #         for i, line in enumerate (f): 

# # #             if (i%10000==0):
# # #                 logging.info ("read {0} reviews".format (i))
# # #             # do some pre-processing and return a list of words for each review text
# # #             yield gensim.utils.simple_preprocess (line)

# # # # read the tokenized reviews into a list
# # # # each review item becomes a serries of words
# # # # so this becomes a list of lists
# # # documents = list (read_input (data_file))
# # # logging.info ("Done reading data file")

# # # read_input()

# # ############# long text vocab

















# # # ##reading corpus like by like, each line contains one sentence
# # # def get_sentences(input_file_pointer):
# # #     while True:
# # #         line = input_file_pointer.readline()
# # #         if not line:
# # #             break
# # #         yield line

# # # def clean_sentence(sentence):
# # #     sentence = sentence.lower().strip()
# # #     sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
# # #     return re.sub(r'\s{2,}', ' ', sentence)

# # # from spacy.lang.en.stop_words import STOP_WORDS
# # # def tokenize(sentence):
# # #     return [token for token in sentence.split() if token not in STOP_WORDS]

# # # from gensim.models.phrases import Phrases, Phraser
# # # def build_phrases(sentences):
# # #     phrases = Phrases(sentences,
# # #                       min_count=5,
# # #                       threshold=7,
# # #                       progress_per=1000)
# # #     return Phraser(phrases)
# # # #save phrases
# # # phrases_model.save('phrases_model.txt')
# # # phrases_model= Phraser.load('phrases_model.txt')

# # # def sentence_to_bi_grams(phrases_model, sentence):
# # #     return ' '.join(phrases_model[sentence])

# # # def sentences_to_bi_grams(n_grams, input_file_name, output_file_name):
# # #     with open(input_file_name, 'r') as input_file_pointer:
# # #         with open(output_file_name, 'w+') as out_file:
# # #             for sentence in get_sentences(input_file_pointer):
# # #                 cleaned_sentence = clean_sentence(sentence)
# # #                 tokenized_sentence = tokenize(cleaned_sentence)
# # #                 parsed_sentence = sentence_to_bi_grams(n_grams, tokenized_sentence)
# # #                 out_file.write(parsed_sentence + '\n')

