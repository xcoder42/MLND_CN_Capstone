'''
对数据集词的处理，生成一个不重复的词典表
'''
# coding: utf-8
import numpy as np
import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer


df_train = pd.read_csv('../../dataset/train.csv')[['question1', 'question2']]
df_test = pd.read_csv('../../dataset/test.csv')[['question1', 'question2']]
df_data = pd.concat([df_train[['question1', 'question2']], df_test[['question1', 'question2']]], axis=0)
        
all_questions = pd.Series(df_data["question1"].tolist() + df_data["question2"].tolist()).unique()
vectorizer = CountVectorizer(lowercase=False, token_pattern="\S+", min_df=100)
vectorizer.fit(all_questions)
vocab = set(vectorizer.vocabulary_.keys())

df_word = pd.DataFrame(data={'word':list(vocab)})
df_word.to_csv('../../dataset/quora_vocab.csv')

    
    
    