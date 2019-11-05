'''
针对于本问题的特征，表现出了图的特性。问题集中有很多的重复（完全一样）问题
问题出现在数据集中的频率越高，是重复问题的概率越大；
问题对中的两个问题，共同邻居越多（已知问题对 a - b，则 a 和 b 互为邻居），是重复问题的概率越大。
'''
# coding: utf-8

import numpy as np
import networkx as nx
import pandas as pd
from itertools import combinations
import sys

def create_question_hash(all_df):
    '''
    将问题映射到编号中
    '''
    all_qs = np.append(all_df.question1, all_df.question2)
    all_qs = pd.DataFrame(all_qs)[0].drop_duplicates()
    all_qs.reset_index(inplace=True, drop=True)
    question_dict = pd.Series(all_qs.index.values, index=all_qs.values).to_dict()
    return question_dict

def make_hash(df_feat, all_df, hash_dict):
    '''
    用编号代替问题，提升效率
    '''
    df_feat["qid1"] = all_df["question1"].map(hash_dict)
    df_feat["qid2"] = all_df["question2"].map(hash_dict)

def make_graph(df_feat):
    '''
    创建图结构
    '''
    g = nx.Graph()
    g.add_nodes_from(df_feat.qid1)
    g.add_nodes_from(df_feat.qid2)
    edges = list(df_feat[['qid1', 'qid2']].to_records(index=False))
    g.add_edges_from(edges)
    # 移除自循环边（自己指向自己）
    g.remove_edges_from(nx.selfloop_edges(g))
    return g

def get_common_neighbors(g, row):
    return len(set(g[row.qid1]).intersection(set(g[row.qid2])))

def make_neighbors_feature(g, df_feat):
    '''
    生成共同邻居特征
    '''
    df_feat['common_neighbors'] = df_feat.apply(lambda row: get_common_neighbors(g, row), axis=1)
    


def make_clique_feature(g, df_feat):
    '''
    clique团大小特征
    '''
    # 初始化clique_size 为1
    df_feat['clique_size'] = 1
    df_feat['temp_index'] = df_feat.apply(lambda x: '{},{}'.format(x.qid1,x.qid2), axis=1)
    df_feat.set_index('temp_index', inplace=True)
    cliques = list(nx.find_cliques(g))
    # 只取包含两个节点以上的
    cliques_tmp = filter(lambda x: len(x) > 2, cliques)
    cliques = list(cliques_tmp)
    cliques = sorted(cliques, key=lambda x: len(x))
    for cli in cliques:
        for q1, q2 in combinations(cli, 2):
            try:
                key1 = '{},{}'.format(q1,q2)
                df_feat.at[key1, 'clique_size'] = len(cli)
            except:
                pass
            try:
                key2 = '{},{}'.format(q2,q1)
                df_feat.at[key2, 'clique_size'] = len(cli)
            except:
                pass

    
def get_kshell_dict(g, df_feat):
    '''
     K-Core 是最大的实体组，所有实体都至少与组中的 k 个其他实体相连
     The k-shell is the subgraph of nodes in the k-core but not in the (k+1)-core
    '''
    MAX_KSHELL = 44
    df_temp = pd.DataFrame(data=g.nodes(), columns=["qid"])
    # 初始化k-shell为1
    df_temp['kshell'] = 1
    for k in range(2, MAX_KSHELL + 1):
        sk = nx.k_shell(g, k=k).nodes()
        df_temp.loc[df_temp.qid.isin(sk), "kshell"] = k
    
    return df_temp.kshell.to_dict()

def make_kshell_feature(g, df_feat):
    kshell_dict = get_kshell_dict(g, df_feat)
    df_feat['kshell_1'] = df_feat.qid1.apply(lambda x: kshell_dict[x])
    df_feat['kshell_2'] = df_feat.qid2.apply(lambda x: kshell_dict[x])


df_train = pd.read_csv('../dataset/train.csv').fillna("")
df_test = pd.read_csv('../dataset/test.csv').fillna("")
len_train = df_train.shape[0]

all_df = pd.concat([df_train[['question1', 'question2']], df_test[['question1', 'question2']]], axis=0)
print('all_df shape: ',all_df.shape)
df_feat = pd.DataFrame()
question_dict = create_question_hash(all_df)
make_hash(df_feat, all_df, question_dict)
print('df_feat shape: ', df_feat.shape)

print('开始生成图结构...')
# 生成图结构
g = make_graph(df_feat)
# print('邻接特征...')
# # 生成图共同邻接特征
# make_neighbors_feature(g, df_feat)
print('cique特征...')
# 生成cique特征
make_clique_feature(g, df_feat)
# print('kshell特征...')
# # 生成kshell特征
# make_kshell_feature(g, df_feat)

df_feat.drop(columns=['qid1','qid2'], inplace=True)
df_feat[:len_train].to_csv('../feature_store/train_feature_graph_cli.csv', index=False)
df_feat[len_train:].to_csv('../feature_store/test_feature_graph_cli.csv', index=False)