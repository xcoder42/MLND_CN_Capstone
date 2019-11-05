# coding: utf-8
'''
fuzz 是一个基于编辑距离（Levenshtein Distance or Edit Distance）算法的库
编辑距离算法：是指两个字符串之间，由一个转成另一个所需的最少编辑操作次数。许
可的编辑操作包括将一个字符替换成另一个字符，插入一个字符，删除一个字符。一般
来说，编辑距离越小，两个串的相似度越大。
'''
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

df_train = pd.read_csv('../dataset/train.csv')
df_test = pd.read_csv('../dataset/test.csv')
len_train = df_train.shape[0]

df_feat = pd.DataFrame()
df_data = pd.concat([df_train[['question1', 'question2']], df_test[['question1', 'question2']]], axis=0)

# qratio: 快速比较两个字串相似度，先对串进行一些预处理（如只保留字母数字，全转小写等），再调用ratio函数
df_feat['fuzz_qratio'] = df_data.apply(lambda row: fuzz.QRatio(str(row['question1']), str(row['question2'])), axis=1)
# wratio: 一个综合性的算法比较，在一般的ratio之后，会根据句长的比例来选择进一步的算法（partial等），使结果更准确
df_feat['fuzz_wratio'] = df_data.apply(lambda row: fuzz.WRatio(str(row['question1']), str(row['question2'])), axis=1)
# partial_ratio: 返回最相似的字串的分数（搜索匹配）
df_feat['fuzz_partial_ratio'] = df_data.apply(lambda row: fuzz.partial_ratio(str(row['question1']), str(row['question2'])), axis=1)
# token_set_ratio: 去掉重复词后全匹配，对顺序不敏感
df_feat['fuzz_token_set_ratio'] = df_data.apply(lambda row: fuzz.token_set_ratio(str(row['question1']), str(row['question2'])), axis=1)
# token_sort_ratio: 排序后全匹配，对顺序不敏感
df_feat['fuzz_token_sort_ratio'] = df_data.apply(lambda row: fuzz.token_sort_ratio(str(row['question1']), str(row['question2'])), axis=1)
# partial_token_set_ratio: 去掉重复词后搜索匹配，对顺序不敏感
df_feat['fuzz_partial_token_set_ratio'] = df_data.apply(lambda row: fuzz.partial_token_set_ratio(str(row['question1']), str(row['question2'])), axis=1)
# partial_token_sort_ratio: 排序后搜索匹配，对顺序不敏感
df_feat['fuzz_partial_token_sort_ratio'] = df_data.apply(lambda row: fuzz.partial_token_sort_ratio(str(row['question1']), str(row['question2'])), axis=1)


df_feat[:len_train].to_csv('../feature_store/train_feature_fuzz.csv', index=False)
df_feat[len_train:].to_csv('../feature_store/test_feature_fuzz.csv', index=False)