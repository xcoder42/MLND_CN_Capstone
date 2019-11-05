'''
由于硬件原因，只能将数据集拆分来计算
'''
# coding: utf-8

import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from w2v_calculate import *

from multiprocessing import Process, Manager
import time


# 多进程任务函数  
def task(model, norm_model, df_part, df_queue):
    df_feat = pd.DataFrame()
    df_feat['order'] = df_part.index.to_list()
    
    # 词移距离特征
    df_feat['glove_wmd'] = df_part.apply(lambda row: word_mover_distance(model, row['question1'], row['question2']), axis=1)
    df_feat['glove_norm_wmd'] = df_part.apply(lambda row: word_mover_distance(norm_model, row['question1'], row['question2']), axis=1)
    print('词移距离特征')
    q1_vectors = np.zeros((df_part.shape[0], 300))
    q2_vectors = np.zeros((df_part.shape[0], 300))
    for i,(q1,q2) in enumerate(zip(df_part.question1, df_part.question2)):
        q1_vectors[i,:] = sentence2vec(model, q1)
        q2_vectors[i,:] = sentence2vec(model, q2)
    print('句子向量计算')
    

    # 各种距离计算
    df_feat['glove_cosine_distance'] = cosine_distance(q1_vectors, q2_vectors)
    df_feat['glove_cityblock_distance'] = cityblock_distance(q1_vectors, q2_vectors)
    df_feat['glove_jaccard_distance'] = jaccard_distance(q1_vectors, q2_vectors)
    df_feat['glove_canberra_distance'] = canberra_distance(q1_vectors, q2_vectors)
    df_feat['glove_euclidean_distance'] = euclidean_distance(q1_vectors, q2_vectors)
    df_feat['glove_minkowski_distance'] = minkowski_distance(q1_vectors, q2_vectors)
    df_feat['glove_braycurtis_distance'] = braycurtis_distance(q1_vectors, q2_vectors)

    # 偏度（skew）和峰度（kurtosis）特征
    df_feat['glove_skew_q1'] = vec_skew(q1_vectors)
    df_feat['glove_skew_q2'] = vec_skew(q2_vectors)
    df_feat['glove_kurtosis_q1'] = vec_kurtosis(q1_vectors)
    df_feat['glove_kurtosis_q2'] = vec_kurtosis(q2_vectors)
    print('距离计算')
    df_queue.put(df_feat)
     

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    df_train = pd.read_csv('../../dataset/train.csv')[['question1', 'question2']]
    df_test = pd.read_csv('../../dataset/test.csv')[['question1', 'question2']]
    df_data = pd.concat([df_train[['question1', 'question2']], df_test[['question1', 'question2']]], axis=0)
    df_data.reset_index(drop=True,inplace=True)
    
    # 加载预训练的GoogleNews词向量模型（权重矩阵）
    model = KeyedVectors.load('../../corpora/glove.wv', mmap='r')
    # l2正则化后
    norm_model = KeyedVectors.load('../../corpora/glove.wv')
    norm_model.init_sims(replace=True)
    
    
    split_num = 8
    # 要使用这个队列，不然在put的时候会阻塞
    df_queue = Manager().Queue(10)
    splits = np.array_split(df_data.index, split_num)
    all_start = time.time()
   
    print("多进程开始······")
    start = time.time()
  
    workers = []
    for part in splits:
        worker = Process(target=task,args=(model, norm_model, df_data.loc[part], df_queue,))
        workers.append(worker)
    
    for worker in workers:
        worker.start()
        
    for worker in workers:
        worker.join()
        

    print("子进程处理结束,时间：", time.time() - start)
    df_feat = pd.DataFrame()
    while not df_queue.empty():
        item_feat = df_queue.get()
        df_feat = pd.concat([df_feat, item_feat], axis=0)
    
    
    # 将feat还原为原始位置
    df_feat = df_feat.sort_values(by='order')

    print("总耗时：", time.time() - all_start)

    df_feat.to_csv('../../feature_store/w2v_glove/w2v_glove_{}.csv'.format(serial), index=False)