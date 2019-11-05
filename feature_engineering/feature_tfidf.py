# coding: utf-8
'''
词频TF（item frequency）:某一给定词语在文本中出现的频率
    TF = 词在文本中出现次数 / 总词数
逆向文件频率IDF（inverse document frequency）：如果包含词条t的文档越少, IDF越大，则说明词条具有很好的类别区分能力
    IDF = log（语料库中文本总数 / （包含该词的文本数）+ 1）
TF-IDF = TF * IDF
'''
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


df_train = pd.read_csv('../dataset/train.csv')
df_test = pd.read_csv('../dataset/test.csv')
len_train = df_train.shape[0]


df_feat = pd.DataFrame()
df_data = pd.concat([df_train[['question1', 'question2']], df_test[['question1', 'question2']]], axis=0)

# 设置英文停用词，避免干扰
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))

# 将问题组合起来形成文档库
questions_txt = pd.Series(
    df_data['question1'].tolist() +
    df_data['question2'].tolist()
).astype(str)

tfidf.fit_transform(questions_txt)

# 分别计算一个文本tfidf值的 总和，均值，长度
tfidf_sum1 = []
tfidf_sum2 = []
tfidf_mean1 = []
tfidf_mean2 = []
tfidf_len1= []
tfidf_len2 = []

for index, row in df_data.iterrows():
    tfidf_q1 = tfidf.transform([str(row.question1)]).data
    tfidf_q2 = tfidf.transform([str(row.question2)]).data
    
    tfidf_sum1.append(np.sum(tfidf_q1))
    tfidf_sum2.append(np.sum(tfidf_q2))
    tfidf_mean1.append(np.mean(tfidf_q1))
    tfidf_mean2.append(np.mean(tfidf_q2))
    tfidf_len1.append(len(tfidf_q1))
    tfidf_len2.append(len(tfidf_q2))

df_feat['tfidf_sum1'] = tfidf_sum1
df_feat['tfidf_sum2'] = tfidf_sum2
df_feat['tfidf_mean1'] = tfidf_mean1
df_feat['tfidf_mean2'] = tfidf_mean2
df_feat['tfidf_len1'] = tfidf_len1
df_feat['tfidf_len2'] = tfidf_len2


df_feat.fillna(0.0)
df_feat[:len_train].to_csv('../feature_store/train_feature_tfidf.csv', index=False)
df_feat[len_train:].to_csv('../feature_store/test_feature_tfidf.csv', index=False)