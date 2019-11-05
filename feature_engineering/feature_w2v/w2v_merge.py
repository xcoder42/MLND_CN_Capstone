# coding: utf-8

import pandas as pd

def merge_gnews(len_train):
    w2v = pd.DataFrame()
    for i in range(10):
        part = pd.read_csv('../../feature_store/w2v_gnews/w2v_gnews_{}.csv'.format(i))
        w2v = pd.concat([w2v, part], axis=0)
    
    w2v = w2v.sort_values(by='order')
    w2v.drop(columns=['order'], inplace=True)
    w2v[:len_train].to_csv('../../feature_store/train_feature_w2v_gnews.csv', index=False)
    w2v[len_train:].to_csv('../../feature_store/test_feature_w2v_gnews.csv', index=False)

def merge_glove(len_train):
    w2v = pd.DataFrame()
    for i in range(10):
        part = pd.read_csv('../../feature_store/w2v_glove/w2v_glove_{}.csv'.format(i))
        w2v = pd.concat([w2v, part], axis=0)
    
    w2v = w2v.sort_values(by='order')
    w2v.drop(columns=['order'], inplace=True)
    w2v[:len_train].to_csv('../../feature_store/train_feature_w2v_glove.csv', index=False)
    w2v[len_train:].to_csv('../../feature_store/test_feature_w2v_glove.csv', index=False)

if __name__ == '__main__':
    len_train = 404290
    #merge_gnews(len_train)
    merge_glove(len_train)