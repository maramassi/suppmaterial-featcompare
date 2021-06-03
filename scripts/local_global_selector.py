import gensim.downloader as api
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from smart_open import smart_open
import os
import pandas as pd
import numpy as np

# all user reviews of all the categories pre-processed in the global cycle of ABAE
all_reviews = pd.read_csv('all_reviews.csv', header=None)

# convert the text in the all_reviews to a list
df = all_reviews.astype(str).apply(lambda x: x.str.lower()).values.tolist()

corpus = [[word for word in document[0].split()] for document in df]

dictionary = Dictionary(corpus)
dictionary.token2id  # Token to Id map

bow_corpus = [dictionary.doc2bow(text) for text in corpus]

model = TfidfModel(bow_corpus, normalize=True)

# creating a new df to add the tf-idf score sum of all the reviews sequencially
tf_idf_info_col = ['tf_idf_sum', 'num_word_rev', 'tf_idf_avr', 'is_global']
tf_idf_info = pd.DataFrame(columns=tf_idf_info_col)

tf_idf_sum_list = []
num_word_rev_list = []
tf_idf_avr_list = []

for doc in model[bow_corpus]:
    sum_value_arr = sum(np.around(freq, decimals=2) for freq in doc)
    sum_value = sum_value_arr[1]
    avr_value = np.around(sum_value / len(doc), decimals=2)

    tf_idf_sum_list.append(np.around(sum_value, decimals=2))
    tf_idf_avr_list.append(avr_value)
    num_word_rev_list.append(len(doc))

tf_idf_info['tf_idf_sum'] = tf_idf_sum_list
tf_idf_info['num_word_rev'] = num_word_rev_list
tf_idf_info['tf_idf_avr'] = tf_idf_avr_list

q1 = tf_idf_info['tf_idf_avr'].quantile([0.25])

tf_idf_info['is_global'] =  tf_idf_info['tf_idf_avr'].apply(lambda x: 1 if x < q1.values else 0)

tf_idf_info.to_csv(r"reviews_tf_idf_info.csv", index=False)