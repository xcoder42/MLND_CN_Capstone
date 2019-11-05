'''
将深度学习网络的输入准备好，以及embedding_matrix
'''
# coding: utf-8

from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

import numpy as np
import pandas as pd
import pickle
import re

from utils import _save, _load, SaveData


np.random.seed(0)
STOP_WORDS = set(stopwords.words('english'))
MAX_SEQUENCE_LENGTH = 30
MIN_WORD_OCCURRENCE = 100
REPLACE_WORD = "cylhope"
EMBEDDING_DIM = 300
EMBEDDING_FILE = '../corpora/glove.wv'

def preprocess(string):
    string = string.lower().replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
        .replace("€", " euro ").replace("'ll", " will").replace("=", " equal ").replace("+", " plus ")
    string = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', string)
    string = re.sub(r"([0-9]+)000000", r"\1m", string)
    string = re.sub(r"([0-9]+)000", r"\1k", string)
    return string

def get_embedding():
    embeddings_index = {}
    f = open(EMBEDDING_FILE, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        if len(values) == EMBEDDING_DIM + 1 and word in top_words:
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def is_numeric(s):
    return any(i.isdigit() for i in s)

def prepare(q):
    new_q = []
    new_cylhope = True
    for w in q.split()[::-1]:
        if w in top_words:
            new_q = [w] + new_q
            new_cylhope = True
        elif w not in STOP_WORDS:
            if new_cylhope:
                new_q = ["cylhope"] + new_q
                new_cylhope = False
        else:
            new_cylhope = True
        if len(new_q) == MAX_SEQUENCE_LENGTH:
            break
    new_q = " ".join(new_q)
    return new_q

def get_prepare(df):
    q1s = np.array([""] * len(df), dtype=object)
    q2s = np.array([""] * len(df), dtype=object)

    for i, (q1, q2) in enumerate(list(zip(df["question1"], df["question2"]))):
        q1s[i] = prepare(q1)
        q2s[i] = prepare(q2)

    return q1s, q2s


# 加载数据集
train = pd.read_csv('../dataset/train.csv')
test = pd.read_csv('../dataset/test.csv')

train["question1"] = train["question1"].fillna("").apply(preprocess)
train["question2"] = train["question2"].fillna("").apply(preprocess)

print("Creating the vocabulary of words occurred more than", MIN_WORD_OCCURRENCE)
all_questions = pd.Series(train["question1"].tolist() + train["question2"].tolist()).unique()
vectorizer = CountVectorizer(lowercase=False, token_pattern="\S+", min_df=MIN_WORD_OCCURRENCE)
vectorizer.fit(all_questions)
top_words = set(vectorizer.vocabulary_.keys())
top_words.add(REPLACE_WORD)

embeddings_index = get_embedding()
print("Words are not found in the embedding:", top_words - embeddings_index.keys())
top_words = embeddings_index.keys()

print("Train questions are being prepared for DNN...")
q1s_train, q2s_train = get_prepare(train)

tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(np.append(q1s_train, q2s_train))
word_index = tokenizer.word_index

train_pad_1 = pad_sequences(tokenizer.texts_to_sequences(q1s_train), maxlen=MAX_SEQUENCE_LENGTH)
train_pad_2 = pad_sequences(tokenizer.texts_to_sequences(q2s_train), maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(train["is_duplicate"])

print("Same steps are being applied for test...")
q1s_test, q2s_test = get_prepare(test)
test_pad_1 = pad_sequences(tokenizer.texts_to_sequences(q1s_test), maxlen=MAX_SEQUENCE_LENGTH)
test_pad_2 = pad_sequences(tokenizer.texts_to_sequences(q2s_test), maxlen=MAX_SEQUENCE_LENGTH)

print("Make Embedding Matrix...")
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print("Save Data...")
save_data = SaveData()
save_data.train_pad_1 = train_pad_1
save_data.train_pad_2 = train_pad_2
save_data.labels = labels
save_data.test_pad_1 = test_pad_1
save_data.test_pad_2 = test_pad_2
save_data.test_ids = test['test_id']
save_data.embedding_matrix = embedding_matrix
save_data.num_words = nb_words
_save('../dataset/glove_embedding_data.pkl', save_data)
