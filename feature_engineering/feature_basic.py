# coding: utf-8
import pandas as pd
import numpy as np
import nltk


df_train = pd.read_csv('../dataset/train.csv')
df_test = pd.read_csv('../dataset/test.csv')
len_train = df_train.shape[0]


df_feat = pd.DataFrame()
# concat axis=0 按行拼接（垂直）
df_data = pd.concat([df_train[['question1', 'question2']], df_test[['question1', 'question2']]], axis=0)


# generate len features
# 通过map函数将文本映射为字符长度（包含空格）
df_feat['q1_len'] = df_data.question1.map(lambda x: len(str(x)))
df_feat['q2_len'] = df_data.question2.map(lambda x: len(str(x)))
# 有效字符个数（不包含空格）
df_feat['q1_char_len'] = df_data.question1.map(lambda x: len(str(x).replace(' ','')))
df_feat['q2_char_len'] = df_data.question2.map(lambda x: len(str(x).replace(' ','')))
# 以空格分割的单词个数
df_feat['q1_word_len'] = df_data.question1.map(lambda x: len(str(x).split()))
df_feat['q2_word_len'] = df_data.question2.map(lambda x: len(str(x).split()))


df_feat['char_len_diff'] = df_feat.q1_char_len - df_feat.q2_char_len
df_feat['word_len_diff'] = df_feat.q1_word_len - df_feat.q2_word_len

def _q_intersection(q1, q2):
    set_q1 = set(str(q1).lower().split())
    set_q2 = set(str(q2).lower().split())
    return len(set_q1.intersection(set_q2))

df_feat['common_words'] = data.apply(lambda x: _q_intersection(x['question1'], x['question2']), axis=1)


df_feat[:len_train].to_csv('../feature_store/train_feature_basic.csv', index=False)
df_feat[len_train:].to_csv('../feature_store/test_feature_basic.csv', index=False)