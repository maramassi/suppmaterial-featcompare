# Import necessary libraries
import gensim.downloader as api
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from smart_open import smart_open
import os
import pandas as pd
import numpy as np

# Load all user reviews of all the categories pre-processed in the global cycle of ABAE
all_reviews = pd.read_csv('all_reviews.csv', header=None)

df = all_reviews.astype(str).apply(lambda x: x.str.lower()).values.tolist()

# Create a corpus for TF-IDF analysis
corpus = [[word for word in document[0].split()] for document in df]

# Create a dictionary mapping tokens to IDs for the corpus
dictionary = Dictionary(corpus)
dictionary.token2id  # Token to Id map

# Create Bag-of-Words (BOW) representation of the corpus and the TF-IDF model
bow_corpus = [dictionary.doc2bow(text) for text in corpus]
model = TfidfModel(bow_corpus, normalize=True)

# Create a new DataFrame to store the TF-IDF scores for each review
tf_idf_info_col = ['tf_idf_sum', 'num_word_rev', 'tf_idf_avr', 'is_global']
tf_idf_info = pd.DataFrame(columns=tf_idf_info_col)

tf_idf_sum_list = []
num_word_rev_list = []
tf_idf_avr_list = []

# Calculate TF-IDF sum, average, and number of words for each review
for doc in model[bow_corpus]:
    sum_value_arr = sum(np.around(freq, decimals=2) for freq in doc)
    sum_value = sum_value_arr[1]
    avr_value = np.around(sum_value / len(doc), decimals=2)

    tf_idf_sum_list.append(np.around(sum_value, decimals=2))
    tf_idf_avr_list.append(avr_value)
    num_word_rev_list.append(len(doc))

# Update the tf_idf_info DataFrame with the calculated values
tf_idf_info['tf_idf_sum'] = tf_idf_sum_list
tf_idf_info['num_word_rev'] = num_word_rev_list
tf_idf_info['tf_idf_avr'] = tf_idf_avr_list

# Calculate the first quartile (25th percentile) of the tf_idf_avr values
q1 = tf_idf_info['tf_idf_avr'].quantile([0.25])

# Set the 'is_global' column to 1 if tf_idf_avr is less than q1, otherwise set it to 0
tf_idf_info['is_global'] = tf_idf_info['tf_idf_avr'].apply(lambda x: 1 if x < q1.values else 0)

tf_idf_info.to_csv(r"reviews_tf_idf_info.csv", index=False)
